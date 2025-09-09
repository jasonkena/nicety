import os
import json
import numpy as np
from typing import List, Tuple, Dict

from tqdm import tqdm
import cloudvolume
from cloudvolume import CloudVolume
from joblib import Parallel, delayed
import igneous.task_creation as tc
from taskqueue import LocalTaskQueue

from nicety.conf import DotDict
from cloudvolume import CloudVolume
from cloudvolume.server import ViewerServerHandler

import threading
import neuroglancer
import socketserver
from urllib.parse import urlparse

# adapted from https://github.com/jasonkena/fiberpipeline/blob/main/fiberpipeline/plot/precomputed.py
# and https://github.com/jasonkena/fiberpipeline/blob/main/fiberpipeline/plot/plot.py

# NOTE: all this expects volumes to either be [Z, Y, X] or [C, Z, Y, X]
# cloudvolume just expects [Z, Y, X, C] internally


def get_free_port(ip: str) -> int:
    # https://stackoverflow.com/a/61685162/10702372
    with socketserver.TCPServer((ip, 0), None) as s:
        return s.server_address[1]


def serve_directory(ip: str, directory: str, port: int):
    def handler(*args, **kwargs):
        return ViewerServerHandler(directory, *args, **kwargs)

    with socketserver.TCPServer((ip, port), handler) as httpd:
        print(f"Serving {directory} at http://localhost:{port}")
        httpd.serve_forever()


def serve_cloudvolume_layers(ip: str, layers: Dict[str, CloudVolume]) -> DotDict:
    layerpaths = {name: cv.layerpath for name, cv in layers.items()}
    common_path = os.path.commonpath(list(layerpaths.values()))
    relative_paths = {
        name: os.path.relpath(path, common_path) for name, path in layerpaths.items()
    }

    port = get_free_port(ip)
    thread = threading.Thread(
        target=serve_directory,
        args=(ip, common_path, port),
        daemon=True,
    )
    thread.start()

    sources = {
        name: f"precomputed://http://{ip}:{port}/{path}"
        for name, path in relative_paths.items()
    }

    return port, thread, sources


def plot_neuroglancer(
    layers: Dict[str, CloudVolume],
    image_kwargs: Dict = {},
    segmentation_kwargs: Dict = {},
):
    ip = "localhost"
    neuroglancer.set_server_bind_address(bind_address=ip)
    viewer = neuroglancer.Viewer()

    port, thread, sources = serve_cloudvolume_layers(ip, layers)
    ports = [port]

    with viewer.txn() as s:
        for name in layers.keys():
            assert layers[name].layer_type in ["image", "segmentation"]

            if layers[name].layer_type == "image":
                s.layers.append(
                    name=name,
                    layer=neuroglancer.ImageLayer(
                        source=sources[name],
                        **image_kwargs,
                    ),
                )
            else:
                cv = layers[name]
                s.layers.append(
                    name=name,
                    layer=neuroglancer.SegmentationLayer(
                        source=sources[name],
                        **segmentation_kwargs,
                    ),
                    # tell NG which segments exist
                    segments=sorted(map(int, cv.image.unique(cv.bounds, cv.mip))),
                )

    print(viewer)
    ports.append(urlparse(str(viewer)).port)
    ports = " ".join(map(str, ports))
    print(f"Ports: {ports}")

    return viewer


def to_precomputed(
    vol: np.ndarray,
    output_layer: str,
    layer_type: str,
    anisotropy: List[float] = [1.0, 1.0, 1.0],
    chunk_size: List[int] = [256, 256, 256],
    downsample_factors: List[List[int]] = [],
    enable_mesh: bool = False,
    mesh_mip: int = 0,
    mesh_merge_magnitude: int = 3,
    enable_skeletons: bool = False,
    n_jobs: int = os.cpu_count(),
):
    """
    Converts a numpy-like array to a cloudvolume precomputed layer.

    Parameters
    ----------
    vol
        either 3D or 4D (C, Z, Y, X)
    output_layer
        directory for precomputed
    layer_type
        either "image" or "segmentation"
    anisotropy
        in same order as chunk_size, e.g. [1, 1, 1] for isotropic
    chunk_size
        size of chunks to write to disk
    downsample_factors
        list of downsample factors for each mip level, e.g. [[1, 1, 1], [2, 2, 2]]
    mesh_mip
        mip level to create mesh for segmentation layer
    mesh_merge_magnitude
        magnitude for merging meshes, only used if layer_type is "segmentation"
    enable_skeletons
        whether to enable skeletons for segmentation layer, if True, will need to be populated after with write_skeletons() or by using igneous skeletonize
    n_jobs
        number of parallel jobs to use for writing chunks
    """
    if enable_skeletons:
        assert layer_type == "segmentation"
    cv = initialize_cloudvolume(
        vol, output_layer, layer_type, anisotropy, chunk_size, enable_mesh, enable_skeletons
    )
    write_cloudvolume(output_layer, vol, chunk_size, n_jobs)
    downsample_cloudvolume(
        output_layer,
        chunk_size=chunk_size,
        downsample_factors=downsample_factors,
        n_jobs=n_jobs,
    )
    if cv.layer_type == "segmentation" and enable_mesh:
        mesh_cloudvolume(
            output_layer,
            mip=mesh_mip,
            merge_magnitude=mesh_merge_magnitude,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
        )

    return cv


def initialize_cloudvolume(
    vol: np.ndarray,
    output_layer: str,
    layer_type: str,
    anisotropy: List[float],
    chunk_size: List[int],
    enable_mesh: bool = False,
    enable_skeletons: bool = False,
):
    """
    generates a cloudvolume precomputed layer from a numpy-like array in [C, Z, Y, X] or [Z, Y, X] format

    Parameters
    ----------
    vol
        either 3D or 4D (C, Z, Y, X)
    output_layer
        directory for precomputed
    layer_type
        either "image or "segmentation"
    anisotropy
        in same order as chunk_size, e.g. [1, 1, 1] for isotropic
    chunk_size
    """
    if enable_skeletons:
        assert (
            layer_type == "segmentation"
        ), "Skeletons can only be enabled for segmentation layers"
    assert layer_type in [
        "image",
        "segmentation",
    ], f"Unsupported layer type: {layer_type}"

    if vol.dtype == np.uint8:
        dtype = "uint8"
    elif vol.dtype == np.uint16:
        dtype = "uint16"
    elif vol.dtype == np.uint32:
        dtype = "uint32"
    elif vol.dtype == np.uint64:
        dtype = "uint64"
    else:
        raise ValueError(f"Unsupported dtype: {vol.dtype}")

    if vol.ndim == 3:
        num_channels = 1
        shape = vol.shape
    elif vol.ndim == 4:
        num_channels = vol.shape[0]
        shape = vol.shape[1:]
    else:
        raise ValueError(f"Unsupported volume shape: {vol.shape}")

    kwargs = {
        "num_channels": num_channels,
        "layer_type": layer_type,
        "data_type": dtype,
        "encoding": "raw",
        "resolution": anisotropy,
        "voxel_offset": [0, 0, 0],
        "chunk_size": chunk_size,
        "volume_size": shape,
    }
    if layer_type == "segmentation":
        if enable_mesh:
            kwargs["mesh"] = "mesh"
        if enable_skeletons:
            kwargs["skeletons"] = "skeletons"

    info = CloudVolume.create_new_info(**kwargs)

    cv = CloudVolume(output_layer, info=info)
    cv.commit_info()

    return cv


def get_chunks(vol_shape: List[int], chunk_size: List[int]):
    assert len(vol_shape) == 3
    assert len(chunk_size) == 3

    chunks = []
    for z in range(0, vol_shape[0], chunk_size[0]):
        for y in range(0, vol_shape[1], chunk_size[1]):
            for x in range(0, vol_shape[2], chunk_size[2]):
                chunks.append(
                    np.s_[
                        z : min(z + chunk_size[0], vol_shape[0]),
                        y : min(y + chunk_size[1], vol_shape[1]),
                        x : min(x + chunk_size[2], vol_shape[2]),
                    ]
                )
    return chunks


def write_cloudvolume_chunk(slice: Tuple[slice], vol: np.ndarray, output_layer: str):
    assert vol.ndim in [3, 4], f"Unsupported volume shape: {vol.shape}"
    if vol.ndim == 4:
        # Convert from [C, Z, Y, X] to [Z, Y, X, C]
        vol = vol[:, slice[0], slice[1], slice[2]].transpose(1, 2, 3, 0)
    else:
        vol = vol[slice[0], slice[1], slice[2]]

    cv = CloudVolume(f"file://{output_layer}")
    cv[slice] = vol


def write_cloudvolume(
    output_layer: str,
    vol: np.ndarray,
    chunk_size: List[int],
    n_jobs: int = os.cpu_count(),
):
    assert vol.ndim in [3, 4], f"Unsupported volume shape: {vol.shape}"
    if vol.ndim == 4:
        shape = vol.shape[1:]
    else:
        shape = vol.shape
    chunks = get_chunks(shape, chunk_size)

    list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(write_cloudvolume_chunk)(c, vol, output_layer) for c in chunks
            ),
            total=len(chunks),
            leave=False,
        )
    )


def downsample_cloudvolume(
    output_layer: str,
    chunk_size: List[int],
    downsample_factors: List[List[int]],
    n_jobs: int = os.cpu_count(),
):
    tq = LocalTaskQueue(parallel=n_jobs)

    layer = f"file://{output_layer}"

    try:
        CloudVolume(layer)
    except:
        print(
            "delete previous mip files, and skeleton files which may have been created with previous mips"
        )
        raise UserWarning("Layer does not exist")

    for i in range(len(downsample_factors)):
        print(f"Downsampling mip {i}")
        downsample_tasks = tc.create_downsampling_tasks(
            layer,
            mip=i,
            num_mips=1,
            factor=tuple(downsample_factors[i]),
            chunk_size=tuple(chunk_size),
        )
        tq.insert(downsample_tasks)
        tq.execute()


def mesh_cloudvolume(
    output_layer: str,
    mip: int = 0,
    merge_magnitude: int = 3,
    chunk_size: List[int] = [256, 256, 256],
    n_jobs: int = os.cpu_count(),
):
    tq = LocalTaskQueue(parallel=n_jobs)
    layer = f"file://{output_layer}"

    tasks = tc.create_meshing_tasks(layer, mip=mip, shape=chunk_size)
    tq.insert(tasks)
    tq.execute()

    tasks = tc.create_mesh_manifest_tasks(layer, magnitude=merge_magnitude)
    tq.insert(tasks)
    tq.execute()


def populate_segment_properties(output_layer: str, label: str = "label"):
    # NOTE: still unsure how to load this info file
    cv = CloudVolume(f"file://{output_layer}")
    unique = cv.image.unique(cv.bounds, cv.mip)
    unique = sorted(map(int, unique))
    unique = list(map(str, unique))

    neuroglancer_data = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": unique,
            "properties": [
                {
                    "id": "label",
                    "type": "label",
                    "values": [label] * len(unique),
                }
            ],
        },
    }

    path = os.path.join(output_layer, "segment_properties", "info")
    os.makedirs(os.path.dirname(path))
    # save as json
    with open(path, "w") as f:
        json.dump(neuroglancer_data, f, indent=4)


def write_skeletons(output_layer: str, skels: List[cloudvolume.Skeleton]):
    """
    Writes skeletons to the cloudvolume layer.
    """
    cv = CloudVolume(f"file://{output_layer}")
    # https://github.com/seung-lab/cloud-volume/issues/540#issue-1233130338
    # NG does not support uint8
    for skel in skels:
        skel.extra_attributes = [
            attr for attr in skel.extra_attributes if attr["data_type"] != "uint8"
        ]
    cv.skeleton.upload(skels)
    cv.skeleton.meta.commit_info()

    info_path = os.path.join(output_layer, "skeletons", "info")
    assert os.path.exists(info_path), f"Skeleton info file does not exist: {info_path}"
    with open(info_path, "r") as f:
        info = json.load(f)
    # https://github.com/seung-lab/cloud-volume/blob/ed2cba49ae15333bf602ba8b359cfd55de1bba98/cloudvolume/datasource/precomputed/skeleton/metadata.py#L117
    assert "vertex_attributes" in info, "Vertex attributes not found in skeleton info"

    info["vertex_attributes"] = [
        attr for attr in info["vertex_attributes"] if attr.get("data_type") != "uint8"
    ]

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

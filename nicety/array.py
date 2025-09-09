import numpy as np
from typing import List, Union, Tuple, Any


def pad_slice(
    vol: np.ndarray, slices: List[Union[slice, int]], mode: str
) -> np.ndarray:
    """
    Given a n-dim volume (np-like array which supports np.s_ slicing) and a slice which may be out of bounds,
    zero-pad the volume to the dimensions of the slice

    the slices have to be in one of the following formats:
        - int (e.g. vol[0])
        - slice(None, None, None) (e.g. vol[:])
        - slice(start, stop, None) (e.g. vol[0:10]) -> the dimensions here will be padded

    output dimension will be
        - 1 if the slice is an int
        - (stop - start) if start and stop are not None
        - vol.shape[i] if start is None and stop is None

    notably, it does not handle ellipsis or np.newaxis

    Parameters
    ----------
    vol
    slices
    mode: np.pad mode
    """
    assert len(vol.shape) == len(
        slices
    ), f"Volume and slices must have the same number of dimensions, given {len(vol.shape)} and {len(slices)}"
    for i, s in enumerate(slices):
        if isinstance(s, int):
            continue
        else:
            assert isinstance(
                s, slice
            ), f"Slice must be an int or a slice, given {type(s)}"
            assert s.step is None, f"Slice step must be None, given {s.step}"
            assert (s.start is None) == (
                s.stop is None
            ), f"Slice start and stop must both be None or not None, given {s.start} and {s.stop}"
            if s.start is not None:
                assert (
                    s.start < s.stop
                ), f"Slice start must be less than stop, given {s.start} and {s.stop}"
                # NOTE: s.start is allowed to be negative
                assert (
                    s.start < vol.shape[i]
                ), f"Slice start must be less than volume shape, given {s.start} and {vol.shape[i]}"
                assert s.stop > 0, f"Slice stop must be greater than 0, given {s.stop}"

    output_shape = []
    for i, s in enumerate(slices):
        if isinstance(s, int):
            output_shape.append(1)
            assert (
                0 <= s < vol.shape[i]
            ), f"Slice {s} is out of bounds for dimension {i}, which has size {vol.shape[i]}"
        else:
            output_shape.append(
                s.stop - s.start if s.start is not None else vol.shape[i]
            )

    input_slices = []
    for i, s in enumerate(slices):
        if isinstance(s, int):
            input_slices.append(s)
        else:
            if s.start is None:
                input_slices.append(slice(None))
            else:
                input_slices.append(slice(max(0, s.start), min(vol.shape[i], s.stop)))

    pad_widths = []
    for i, s in enumerate(slices):
        if isinstance(s, int) or s.start is None:
            pad_widths.append((0, 0))
        else:
            pad_widths.append(
                (
                    max(0, -s.start),
                    max(0, s.stop - vol.shape[i]),
                )
            )

    # so if scalar is indexed i.e. np.arange(5)[0], shape will be [1] instead of ()
    output = np.zeros(output_shape, dtype=vol.dtype)
    output[:] = np.pad(vol[tuple(input_slices)], pad_widths, mode=mode)

    return output


def reshape_list(data: List[Any], shape: Tuple[int, ...]) -> List[Any]:
    """
    Reshapes a list of data into a nested list of the given shape

    Parameters
    ----------
    data
    shape
    """
    assert len(shape) > 0
    if len(shape) == 1:
        assert len(data) == shape[0]
        return data
    else:
        num_elements = np.prod(shape[1:])
        return [
            reshape_list(data[i * num_elements : (i + 1) * num_elements], shape[1:])
            for i in range(shape[0])
        ]


def smallest_dtype(array: Union[np.ndarray, List[np.ndarray]]) -> np.dtype:
    """
    Returns the smallest dtype that can fit all values in the array.

    Args:
        array: numpy array, array-like, or list of numpy arrays of integral type

    Returns:
        numpy dtype: smallest dtype that can fit all values
    """
    if isinstance(array, np.ndarray):
        array = np.asarray(array)
        assert np.issubdtype(array.dtype, np.integer), f"Array must be of integral type, got {array.dtype}"
        min_val = np.min(array)
        max_val = np.max(array)
    else:
        arrays = [np.asarray(arr) for arr in array]
        for arr in arrays:
            assert np.issubdtype(arr.dtype, np.integer), f"Array must be of integral type, got {arr.dtype}"
        min_val = min(np.min(arr) for arr in arrays)
        max_val = max(np.max(arr) for arr in arrays)

    # If all values are non-negative, use unsigned types
    if min_val >= 0:
        if max_val <= np.iinfo(np.uint8).max:
            return np.uint8
        elif max_val <= np.iinfo(np.uint16).max:
            return np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            return np.uint32
        else:
            return np.uint64

    # If there are negative values, use signed types
    else:
        if (
            np.iinfo(np.int8).min <= min_val <= np.iinfo(np.int8).max
            and max_val <= np.iinfo(np.int8).max
        ):
            return np.int8
        elif (
            np.iinfo(np.int16).min <= min_val <= np.iinfo(np.int16).max
            and max_val <= np.iinfo(np.int16).max
        ):
            return np.int16
        elif (
            np.iinfo(np.int32).min <= min_val <= np.iinfo(np.int32).max
            and max_val <= np.iinfo(np.int32).max
        ):
            return np.int32
        else:
            return np.int64

import copy
import argparse
from omegaconf import OmegaConf
from typing import List, Optional, Union


class DotDict(dict):
    # modified from https://stackoverflow.com/a/13520518/10702372
    """
    A dictionary that supports dot notation as well as dictionary access notation.
    Usage: d = DotDict() or d = DotDict({'val1':'first'})
    Set attributes: d.val2 = 'second' or d['val2'] = 'second'
    Get attributes: d.val2 or d['val2']

    NOTE: asserts that dictionary does not contain tuples (YAML)
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = self._convert_list(value)
            self[key] = value

    def _convert_list(self, lst):
        new_list = []
        for item in lst:
            if isinstance(item, dict):
                new_list.append(DotDict(item))
            elif isinstance(item, list):
                new_list.append(self._convert_list(item))
            else:
                new_list.append(item)
        return new_list

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return DotDict(dict(self))

    def __deepcopy__(self, memo):
        return DotDict({key: copy.deepcopy(value, memo) for key, value in self.items()})


def dotdict_to_dict(d: DotDict) -> dict:
    """
    Convert a potentially nested DotDict to a regular dictionary.
    """
    if isinstance(d, DotDict):
        return {k: dotdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [dotdict_to_dict(item) for item in d]
    else:
        return d


def get_conf(configs: Optional[Union[str, List[str]]] = None) -> DotDict:
    if configs is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c",
            "--config",
            action="append",
            help="List of configuration files.",
            required=True,
        )
        args = parser.parse_args()

        confs = [OmegaConf.load(c) for c in args.config]
    else:
        if isinstance(configs, str):
            configs = [configs]
        confs = [OmegaConf.load(c) for c in configs]

    conf = OmegaConf.merge(*confs)

    # cast to dictionary, because hash of OmegaConf fields depend on greater object
    conf = OmegaConf.to_container(conf, resolve=True)
    assert isinstance(conf, dict), "conf must be a dictionary"
    # allow dot access
    conf = DotDict(conf)

    return conf

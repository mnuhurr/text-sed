from pathlib import Path

import json
import yaml
import pickle

from typing import Union, Any


def read_json(filename: Union[str, Path]) -> Any:
    with Path(filename).open('rt') as file:
        data = json.load(file)
        return data


def write_json(filename: Union[str, Path], data: Any):
    with Path(filename).open('wt') as file:
        json.dump(data, file)


def read_yaml(filename: Union[str, Path]) -> Any:
    with Path(filename).open('rt') as file:
        data = yaml.safe_load(file)
        return data


def write_yaml(filename: Union[str, Path], data: Any):
    with Path(filename).open('wt') as file:
        yaml.safe_dump(data, file)


def read_pickle(filename: Union[str, Path]) -> Any:
    with Path(filename).open('rb') as file:
        data = pickle.load(file)
        return data


def write_pickle(filename: Union[str, Path], data: Any):
    with Path(filename).open('wb') as file:
        pickle.dump(data, file)

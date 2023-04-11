import random
import numpy as np
import torch

from typing import Optional, Union, Any, Tuple, List


def get_or_sample_uniform(value: Optional[Union[Any, Tuple[Any, Any]]],
                          size: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return np.random.uniform(value[0], value[1], size=size)
    if size is None:
        return value
    return np.repeat(value, size)


def get_or_sample_choice(value: Optional[Union[Any, Tuple[Any, ...], List[Any]]]) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple) or isinstance(value, list):
        return random.choice(value)
    return value


def get_or_sample_bool(value: Optional[Union[str, bool]]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        return random.choice([True, False])
    return value


def get_or_sample_int(value: Union[int, Tuple[int, int]]) -> int:
    if isinstance(value, tuple):
        return np.random.randint(value[0], value[1] + 1)
    return value


def random_angle(size: Optional[Union[int, Tuple[int, ...]]] = None):
    return np.random.uniform(-np.pi, np.pi, size)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

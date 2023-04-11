import torch.nn as nn

from typing import Callable
from nip import nip


@nip
def get_activation(name: str) -> Callable:
    if name is None or name == "none":
        return nn.Identity
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unknown activation {name}")

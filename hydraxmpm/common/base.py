import dataclasses
from typing import Self

import equinox as eqx

# from typeguard import typechecked
from jaxtyping import Array, Float, jaxtyped


class Base(eqx.Module):
    """Base class contains. setup and replace functions"""

    def __init__(self: Self, **kwargs) -> Self:
        pass

    def replace(self: Self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)

    def test(self, key: str):
        print(type(key))

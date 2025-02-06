import dataclasses
from typing import Self

import equinox as eqx

# from typeguard import typechecked
from jaxtyping import Array, Float, jaxtyped

# from typeguard import typechecked as typechecker
from beartype import beartype as typechecker


# Type-check a function
# @jaxtyped(typechecker=typechecker)
class Base(eqx.Module):
    """Base class contains. setup and replace functions"""

    def __init__(self: Self, **kwargs) -> Self:
        pass

    def replace(self: Self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)

    def test(self, key: str):
        print(type(key))

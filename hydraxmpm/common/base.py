# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import dataclasses
from typing import Self, Optional, Dict

import equinox as eqx

# from typeguard import typechecked
from jaxtyping import Array, Float, jaxtyped


class Base(eqx.Module):
    """Base class contains. setup and replace functions"""

    name: Optional[str] = eqx.field(static=True, default=None)
    other: Optional[Dict] = eqx.field(static=True, default=None)
    error_check: bool = eqx.field(static=True, default=False)

    def __init__(self: Self, **kwargs) -> Self:
        self.name = kwargs.get("name", None)
        self.other = kwargs.get("other", None)
        self.error_check = kwargs.get("error_check", False)

    def replace(self: Self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)

    def test(self, key: str):
        print(type(key))

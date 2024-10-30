from functools import partial
from typing import Tuple
from typing_extensions import Self, Optional
from ..config.mpm_config import MPMConfig

import chex
import jax
import jax.numpy as jnp

from jax.sharding import Sharding
import jax

import equinox as eqx


class Solver(eqx.Module):
    
    config: MPMConfig = eqx.field(static=True)
    
    def __init__(self,config: MPMConfig):
        self.config = config
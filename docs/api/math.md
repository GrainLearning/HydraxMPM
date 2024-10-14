# Math utils

Helper functions are provided for basic mathematical operations. These functions can be called from `utils.math_helpers` or the base `pymudokon` package.

**Example usage:**

This example demonstrates how to calculate the pressure from a stress tensor and a stack of stress tensors. Most of the functions have a vectorized version that can be used to calculate the values for a stack of tensors (denoted by `_stack`).

```python

import pymudokon as pm
import jax.numpy as jnp

# Cauchy stress tensor
stress = -jnp.eye(3)*1000 # shape (3, 3)

pressure = pm.get_pressure(stress) # 1000 [Pa]

# Stack (array) of stress tensors
stress_stack = jnp.stack([stress, stress]) # shape (2, 3, 3)

pressure_stack = pm.get_pressure_stack(stress_stack) # [1000,1000] [Pa] and shape (2,)

```

::: utils.math_helpers
raise NotImplementedError("This module is not implemented yet.")


# Create debug routine to check if all arrays are similar particle shapes, 
# interactions and num particle same
# 
# # def debug_particles(
#         step: jnp.int32,
#         particles: pm.Particles,
#         stress_limit: jnp.float32 = 1e6,
#     ):
#     """
#     First challenge is to narow down the iteration.

#     Second challenge is the find function causing the error.

#     Third challenge is the find the memory location of the error.
#     """

#     # Check out of bounds
#     positions = usl.particles.positions
#     out_of_bounds = jnp.any(
# jnp.logical_or(positions < nodes.origin, positions > nodes.end))
#     if out_of_bounds:
#         Exception(f"Instability detected: Particles out of bounds at step {step}")

#     # Check for NaN or Inf values
#     if jnp.any(jnp.isnan(particles.stresses))
# or jnp.any(jnp.isinf(particles.stresses)):
#         raise Exception(f"Instability detected: NaN or Inf value in stress {step}")

#     # Check for extreme values
#     if jnp.max(jnp.abs(particles.stresses)) > stress_limit:
#         raise Exception("Instability detected: Stress exceeds limit")

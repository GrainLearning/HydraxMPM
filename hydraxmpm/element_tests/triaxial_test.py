import jax.numpy as jnp
import equinox as eqx
from .driver import ElementTestDriver
import jax
from typing import Optional
from jaxtyping import Float, Array


class TriaxialTest(eqx.Module):
    solver: ElementTestDriver
    confine: float | Float[Array, ""]
    axial_rate: float | Float[Array, "..."]
    dt: float | Float[Array, "..."]
    is_undrained: bool = (False,)
    num_steps: Optional[int] = 2000
    num_newton_iters: int = 20
    stride: int = 1
    start_static: bool = False
    """
    Positive axial strain means compression
    
    """

    @property
    def axial_strain_stack(self):
        # Handle scalar vs array case for strain calculation
        if hasattr(self.dt, "shape") and self.dt.shape:
            # Array case: Cumulative sum
            incr_strain = self.axial_rate * self.dt
            return jnp.cumsum(incr_strain)[:: self.stride]
        else:
            # Scalar case (legacy)
            return jnp.linspace(
                0, self.axial_rate * self.dt * self.num_steps, self.num_steps
            )[:: self.stride]

    def run(self, mp_init, law_init):

        def axial_strain_input(idx):
            rate = (
                self.axial_rate[idx]
                if (hasattr(self.axial_rate, "ndim") and self.axial_rate.ndim > 0)
                else self.axial_rate
            )
            d_t = (
                self.dt[idx]
                if (hasattr(self.dt, "ndim") and self.dt.ndim > 0)
                else self.dt
            )

            return rate, d_t

        def scan_outer(carry, idx):
            mp, law = carry

            curr_rate, curr_dt = axial_strain_input(idx)

            def get_L_drained():
                def residual(lateral_rate):
                    
                    L = jnp.zeros((1, 3, 3))
                    L = L.at[0, 0, 0].set(lateral_rate)  # L_11
                    L = L.at[0, 1, 1].set(lateral_rate)  # L_22
                    L = L.at[0, 2, 2].set(curr_rate)  # L_33

       
                    mp_out, _ = self.solver.step(mp, law, L, curr_dt)
   
 
                    sigma_xx = mp_out.stress_stack[0, 0, 0]

                    return sigma_xx - self.confine

                def step_fn(carry, _):
                    x = carry

  
                    R = residual(x)
                    J = jax.grad(residual)(x)

                    dx = -R / J

                    # Damping (0.8 safe?... could be a parameter?)
                    x_new = x + 0.8 * dx


                    is_bad = jnp.any(jnp.isnan(x_new)) | jnp.any(jnp.isinf(x_new))
                    x_new = jnp.where(is_bad, x, x_new)

                    return x_new, None

                x_final, _ = jax.lax.scan(
                    step_fn,
                    0.0,  # initial guess for radial rate
                    None,
                    length=self.num_newton_iters,
                )

                # Run the step for real with the optimized rate
                L_final = jnp.zeros((1, 3, 3))
                L_final = L_final.at[0, 0, 0].set(x_final)
                L_final = L_final.at[0, 1, 1].set(x_final)
                L_final = L_final.at[0, 2, 2].set(curr_rate)
                return L_final

            def get_L_undrained():
                lateral_rate = -0.5 * curr_rate

                L_final = jnp.zeros((1, 3, 3))
                L_final = L_final.at[0, 0, 0].set(lateral_rate)
                L_final = L_final.at[0, 1, 1].set(lateral_rate)
                L_final = L_final.at[0, 2, 2].set(curr_rate)
                return L_final

            L_next = jax.lax.cond(self.is_undrained, get_L_undrained, get_L_drained)
            mp_next, law_next = self.solver.step(mp, law, L_next, curr_dt)

            return (mp_next, law_next), (mp_next, law_next)  # carry, output


        # This runs 'stride' steps and returns the final state of that segment
        def static_run(carry, idx):
            if self.start_static:
                return jax.lax.cond(
                    idx == 0,
                    lambda: (carry, carry),
                    lambda: scan_outer(carry, idx),
                )
            return scan_outer(carry, idx)

        def run_inner(carry, indices_chunk):


            segment_final_state, _ = jax.lax.scan(static_run, carry, indices_chunk)


            return segment_final_state, segment_final_state

        num_steps = (
            len(self.axial_rate)
            if (hasattr(self.axial_rate, "ndim") and self.axial_rate.ndim > 0)
            else self.num_steps
        )
        all_indices = jnp.arange(num_steps)

        num_snapshots = num_steps // self.stride
        reshaped_indices = all_indices.reshape((num_snapshots, self.stride))


        _, trajectory = jax.lax.scan(run_inner, (mp_init, law_init), reshaped_indices)

        if mp_init.num_points == 1:

            def _squeeze_axis1(x):
                try:
                    if hasattr(x, "ndim") and x.ndim >= 2 and x.shape[1] == 1:
                        return jnp.squeeze(x, axis=1)
                except Exception:
                    pass
                return x

            trajectory = jax.tree.map(_squeeze_axis1, trajectory)

        return trajectory

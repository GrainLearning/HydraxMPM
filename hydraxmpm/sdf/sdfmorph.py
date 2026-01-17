
import equinox as eqx

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxtyping import Float, Array

from .sdfobject import SDFObjectBase,SDFObjectState

from ..utils.math_helpers import safe_norm

import jax


class MorphSDFState(SDFObjectState):
    """
    State for a morphing object.
    blend_factor: 0.0 = Shape A, 1.0 = Shape B
    """

    blend_factor: float | Float[Array, ""]
    blend_rate: Float[Array, ""]

class MorphSDF(SDFObjectBase):
    """
    Linearly interpolates between two SDFs.

    Usage:
        morph = MorphSDF(start_shape=BoxSDF(...), end_shape=SphereSDF(...))
        state = morph.create_state(center, rotation, blend_factor=0.5)
    """

    shape_a: SDFObjectBase
    shape_b: SDFObjectBase

    def __init__(
        self, start_shape: SDFObjectBase, end_shape: SDFObjectBase
    ):
        self.shape_a = start_shape
        self.shape_b = end_shape

    def create_state(
        self,
        center_of_mass,
        rotation=None,
        blend_factor: float = 0.0,
        blend_rate: float = 0.0,
        velocity=None,
        angular_velocity=None,
    ) -> MorphSDFState:
        # call parent to get base fields
        base = super().create_state(
            center_of_mass, velocity, angular_velocity, rotation
        )

        return MorphSDFState(
            center_of_mass=base.center_of_mass,
            rotation=base.rotation,
            velocity=base.velocity,
            angular_velocity=base.angular_velocity,
            blend_factor=jnp.asarray(blend_factor),
            blend_rate=jnp.asarray(blend_rate)
        )

    def signed_distance_local(
        self, state, p_local: Float[Array, "dim"]
    ) -> Float[Array, ""]:

        d_a = self.shape_a.signed_distance_local(state, p_local)
        d_b = self.shape_b.signed_distance_local(state, p_local)

        # 2. Linear Interpolation
        # t=0 -> d_a, t=1 -> d_b
        t = jnp.clip(state.blend_factor, 0.0, 1.0)

        return (1.0 - t) * d_a + t * d_b

    def get_velocity(
        self, sdf_state: SDFObjectState, pos_world: Float[Array, "dim"], dt
    ) -> Float[Array, "dim"]:
        """
        Calculates kinematic velocity for a single point.
        """
        # Note everything is calculated in WORLD coordinates

        p_local = pos_world - sdf_state.center_of_mass

        # v = v_lin + w x r
        # Handle 2D vs 3D cross product
        if p_local.shape[0] == 2:
            # 2D Cross product: omega is scalar, r is vector
            # [-w * ry, w * rx]
            cross = jnp.array(
                [
                    -sdf_state.angular_velocity * p_local[1],
                    sdf_state.angular_velocity * p_local[0],
                ]
            )
        else:
            # 3D in WORLD frame
            cross = jnp.cross(sdf_state.angular_velocity, p_local)

        # Morph velocity via AD
        #  Spatial Gradient (d_phi / d_x) -> Normal
        # Note: We use the WORLD signed_distance, which includes rotation/translation logic
        grad_spatial_fn = jax.grad(self.signed_distance, argnums=1)
        spatial_grad = grad_spatial_fn(sdf_state, pos_world)

        # Parameter Gradient (d_phi / d_state)
        # This returns a MorphSDFState object full of gradients
        grad_state_fn = jax.grad(self.signed_distance, argnums=0)
        state_grads = grad_state_fn(sdf_state, pos_world)

        # Extract sensitivity to blend_factor
        dphi_dblend = state_grads.blend_factor

        # dphi/dt = (dphi/dblend) * (dblend/dt)
        dphi_dt = dphi_dblend * sdf_state.blend_rate

        # prevent artificial suction
        dphi_dt_clamped = jnp.minimum(dphi_dt, 0.0) 

        # C. Compute Morph Velocity Vector
        # v_morph = - ( dphi_dt / |grad phi|^2 ) * grad phi
        norm_sq = jnp.sum(spatial_grad**2)

        # Epsilon prevents NaN deep inside object
        v_morph = -(dphi_dt_clamped / (norm_sq + 1e-12)) * spatial_grad

        v_body = sdf_state.velocity + cross + v_morph

        return v_body



def update_morph(sim_time, sdf_state):
    speed_factor = 5.0

    def compute_blend(t):
        omega = jnp.pi * speed_factor
        return 0.5 * (1.0 + jnp.sin(t * omega))

    b_factor, b_rate = jax.value_and_grad(compute_blend)(sim_time)

    # 3. Update BOTH fields in the state
    return eqx.tree_at(
        lambda s: (s.blend_factor, s.blend_rate), sdf_state, (b_factor, b_rate)
    )




class ChainMorphSDF(SDFObjectBase):
    shapes: list[SDFObjectBase]

    def __init__(self, shapes: list[SDFObjectBase]):
        self.shapes = shapes

    def create_state(self, center_of_mass, rotation=None, blend_factor=0.0, blend_rate=0.0, velocity=None, angular_velocity=None):
        base = super().create_state(center_of_mass, velocity, angular_velocity, rotation)
        return MorphSDFState(
            center_of_mass=base.center_of_mass,
            rotation=base.rotation,
            velocity=base.velocity,
            angular_velocity=base.angular_velocity,
            blend_factor=jnp.asarray(blend_factor),
            blend_rate=jnp.asarray(blend_rate)
        )

    def signed_distance_local(self, state, p_local):
        # Compute distances for all shapes
        dists = jnp.stack([s.signed_distance_local(state, p_local) for s in self.shapes])
        
        T = state.blend_factor
        num_segments = len(self.shapes) - 1
        T = jnp.clip(T, 0.0, num_segments)
        
        # Identify segment
        idx = jnp.minimum(jnp.floor(T).astype(int), num_segments - 1)
        t = T - idx
        
        d_start = dists[idx]
        d_end   = dists[idx + 1]
        
        return (1.0 - t) * d_start + t * d_end

    def get_velocity(self, sdf_state, pos_world, dt):
        p_local = pos_world - sdf_state.center_of_mass
        if p_local.shape[0] == 2:
            cross = jnp.array([-sdf_state.angular_velocity * p_local[1], sdf_state.angular_velocity * p_local[0]])
        else:
            cross = jnp.cross(sdf_state.angular_velocity, p_local)

        grad_spatial_fn = jax.grad(self.signed_distance, argnums=1)
        spatial_grad = grad_spatial_fn(sdf_state, pos_world)

        grad_state_fn = jax.grad(self.signed_distance, argnums=0)
        state_grads = grad_state_fn(sdf_state, pos_world)

        dphi_dblend = state_grads.blend_factor
        dphi_dt = dphi_dblend * sdf_state.blend_rate
        dphi_dt_clamped = jnp.minimum(dphi_dt, 0.0) 

        norm_sq = jnp.sum(spatial_grad**2)
        v_morph = -(dphi_dt_clamped / (norm_sq + 1e-12)) * spatial_grad

        return sdf_state.velocity + cross + v_morph
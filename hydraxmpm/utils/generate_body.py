import jax
import jax.numpy as jnp
import numpy as np


def generate_particles_in_sdf(
    *,
    sdf_obj,
    bounds_min=None,
    bounds_max=None,
    num_particles=None,  # For 'random' mode
    sdf_state=None,
    key=None,
    center_of_mass=None,
    rotation=None,
    shapefunction="cubic",
    mode="uniform",
    cell_size=None,  # For 'regular'/'gauss_legendre' modes
    ppc=None,  # For 'regular'/'gauss_legendre' modes
):
    """
    Generates particles inside the SDF using Rejection Sampling.

    mode options:
    "rejection" - rejection sampling monte carlo
    "regular" - uniform grid sampling (requires cell_size and ppc)
    "gauss_legendre - gauss legendre quadrature sampling (requires cell_size and ppc)
    """
    dim = bounds_min.shape[0]

    if sdf_state is None:
        if center_of_mass is None:
            center_of_mass = (bounds_min + bounds_max) / 2.0
        sdf_state = sdf_obj.create_state(
            center_of_mass=center_of_mass, rotation=rotation
        )

    candidate_point_stack = None
    if mode == "random":
        if num_particles is None or key is None:
            raise ValueError("Mode 'random' requires 'num_particles' and 'key'.")

        key, subkey = jax.random.split(key)
        candidate_point_stack = jax.random.uniform(
            subkey, (num_particles, dim), minval=bounds_min, maxval=bounds_max
        )
    elif mode in ["regular", "gauss_legendre"]:

        # Create integer grid indices that cover the bounding box
        start_idx = jnp.floor(bounds_min / cell_size).astype(int)
        end_idx = jnp.ceil(bounds_max / cell_size).astype(int)

        # Create grid coordinates in grid index space
        id_ranges = [jnp.arange(s, e) for s, e in zip(start_idx, end_idx)]
        meshes = jnp.meshgrid(*id_ranges, indexing="ij")

        grid_origins_flat = jnp.stack(meshes, axis=-1).reshape(-1, dim) * cell_size
        offsets = None

        if mode == "regular":

            # Do linear subdivision: e.g., ppc=4 (2D) -> 2x2 grid inside cell
            pts_per_dim = int(np.ceil(ppc ** (1 / dim)))

            # Linspace from 0 to 1 to shift center points
            # e.g., 2 points -> [0.25, 0.75]
            lin = jnp.linspace(0, 1, pts_per_dim * 2 + 1)[1::2]

            offset_meshes = jnp.meshgrid(*([lin] * dim), indexing="ij")
            offsets = jnp.stack(offset_meshes, axis=-1).reshape(-1, dim)

        # Perform a broadcast add
        # grid_origins_flat has shape (N_cells, 1, dim)
        # offsets has shape  (1, N_offsets, dim)
        # gives shifted origins and we reshape (N_cells * N_offsets, dim)
        candidate_point_stack = (
            grid_origins_flat[:, None, :] + offsets[None, :, :]
        ).reshape(-1, dim)

    dists = sdf_obj.get_signed_distance_stack(sdf_state, candidate_point_stack)
    mask = dists < 0.0

    # Filter and return points inside the SDF
    
    valid_points = candidate_point_stack[mask]

    return valid_points

import zarr
import numcodecs
import jax.numpy as jnp
import equinox as eqx
import os
import jax
import numpy as np
import json

def save_compressed_sim(
    state,
    path="simulation_results.zarr",
    convert_to_float16=True,
    save_only_mp_state=True,
    mp_idx=0,
    metadata=None,
):
    """
    Saves a nested Equinox/JAX state to a Zarr store using Format 2
    to ensure compatibility with numcodecs and BITSHUFFLE.
    """

    ###########################################################################
    # Set compressor
    ###########################################################################

    # BITSHUFFLE is critical for high-entropy MPM data
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE
    )

    ###########################################################################
    # Setup Zarr store
    ###########################################################################

    # note: format 2 is required to avoid numcodecs TypeError with JAX
    root = zarr.open(path, mode="w", zarr_format=2)

    if metadata is not None:
        # Ensure metadata is JSON serializable (convert JAX arrays to lists)
        def convert_jax(obj):
            if isinstance(obj, (jnp.ndarray, np.ndarray)):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_jax(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_jax(i) for i in obj]
            return obj

        serializable_meta = convert_jax(metadata)

        # Save parameters.json inside the Zarr directory
        json_path = os.path.join(path, "parameters.json")
        with open(json_path, "w") as f:
            json.dump(serializable_meta, f, indent=4)

    # partition state to only save JAX arrays
    # HydraxMPM state is anyway JAX arrays
    # logic is stored in logic containers
    dynamic_state, _ = eqx.partition(state, eqx.is_array)

    if save_only_mp_state:
        dynamic_state = state.world.material_points[mp_idx]

        # note we might have stress overflow so converting to kPa

        dynamic_state = eqx.tree_at(
            lambda s: s.stress_stack,
            dynamic_state,
            dynamic_state.stress_stack
            / 1000.0,  # Keep only the first snapshot dimension
        )

    ###########################################################################
    # Convert JAX PyTree Leaf to Zarr
    ###########################################################################

    def leaf_to_zarr(path_tuple, array):
        if array is None:
            return

        if jnp.issubdtype(array.dtype, jnp.floating) and convert_to_float16:
            array = array.astype(jnp.float16)

        key = "/".join(
            [
                str(p)
                .replace("GetAttrKey(name='", "")
                .replace("')", "")
                .replace("SequenceKey(idx=", "")
                .replace(")", "")
                .replace("DictKey(key='", "")
                .replace("')", "")
                for p in path_tuple
            ]
        )

        if array.ndim >= 3:
            # We want: 1 Batch, 1 Snapshot, All Particles
            # For everything else (like the 3x3 stress tensor), use the full dim size
            chunks = (1, 1, array.shape[2]) + array.shape[3:]
        else:
            # For 1D or 2D arrays (like global time or sim_id),
            # let Zarr handle it or use the full shape
            chunks = array.shape

        # Create and fill the dataset
        z_arr = root.create_dataset(
            key,
            shape=array.shape,
            chunks=chunks,
            dtype=array.dtype,
            compressor=compressor,
            overwrite=True,
        )
        # Use np.asarray to ensure JAX arrays are converted correctly for the Zarr writer
        z_arr[:] = jax.device_get(array)

    jax.tree_util.tree_map_with_path(leaf_to_zarr, dynamic_state)

    print(f"Simulation state saved to {path} (Zarr v2 format)")

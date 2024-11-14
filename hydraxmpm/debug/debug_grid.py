import jax.numpy as jnp
import matplotlib.pyplot as plt

from ..partition.grid_stencil_map import GridStencilMap


def give_grid_ids_plot(config, position_stack, fig_ax=None):
    """Debug function to plot grid ids (hashes), num particles and particle ids in 2D."""
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(20, 20))

    grid = GridStencilMap(config)

    grid = grid.partition(position_stack)

    cell_pos_stack = jnp.array(jnp.unravel_index(grid.cell_id_stack, config.grid_size))

    X_stack, Y_stack = cell_pos_stack

    ax.scatter(X_stack, Y_stack, color="blue", marker="s")

    for i, txt in enumerate(grid.cell_id_stack):
        neigh_id_list = []
        num_neigh = grid.cell_count_stack.at[i].get()
        neigh_start = grid.cell_start_stack.at[i].get()
        for ci in range(num_neigh):
            index = neigh_start + ci
            p_id = int(grid.point_id_sorted_stack.at[index].get())

            neigh_id_list.append(p_id)

        label = f"{i}:{num_neigh}:{neigh_id_list}"
        ax.annotate(label, (X_stack[i], Y_stack[i]))

    return fig, ax


def give_point_ids_plot(config, position_stack, fig_ax=None):
    """Debug function to plot particle ids and hashes in 2D."""
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(20, 20))

    grid = GridStencilMap(config)

    grid = grid.partition(position_stack=position_stack)

    X_stack, Y_stack = position_stack.T * config.inv_cell_size

    ax.scatter(X_stack, Y_stack, color="red", marker="o")

    for i in range(grid.point_id_stack.shape[0]):
        p_id = jnp.arange(grid.num_points).at[i].get()
        p_hash = grid.point_hash_stack.at[i].get()

        label = f"{p_hash}:{p_id}"
        ax.annotate(label, (X_stack[i], Y_stack[i]))

    return fig, ax

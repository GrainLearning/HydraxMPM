"""Pressure-projected USL-AFLIP solver for incompressible MPM materials."""

import itertools
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..constitutive_laws.mu_i_rheology import (
    MuI_IC,
    MuIIncompressibleState,
)
from ..forces.force import Force
from ..forces.sdf_collider import SDFCollider
from ..grid.grid import GridDomain
from ..sdf.sdfobject import SDFObjectBase
from .coupling import BodyCoupling
from .usl_asflip import USLAFLIP


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


class USLIncompressibleAFLIP(USLAFLIP):
    """AFLIP momentum update followed by a cell-pressure projection.

    Pressure is piecewise constant at cell centers and velocities remain at
    grid nodes (a semi-staggered MPM grid).  The projection is the weighted
    minimum-energy correction whose cell divergence is zero.  SDF-normal
    velocity degrees of freedom inside solid boundaries are held fixed.

    This initial implementation intentionally supports one incompressible
    material/grid coupling per solver.  That restriction prevents an invalid
    projection across unrelated materials or grids.
    """

    cell_node_hashes: Int[Array, "num_pressure_cells num_cell_nodes"]
    cell_node_gradients: Float[Array, "num_cell_nodes dim"]
    edge_left: Int[Array, "num_pressure_edges"]
    edge_right: Int[Array, "num_pressure_edges"]
    cell_grid_size: Tuple[int, ...] = eqx.field(static=True)
    projection_mass_cutoff: float = eqx.field(static=True)
    projection_regularization: float = eqx.field(static=True)
    pressure_stabilization: float = eqx.field(static=True)
    projection_iterations: int = eqx.field(static=True)
    nonnegative_pressure: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        grid_domains: Tuple[GridDomain, ...],
        constitutive_laws: Tuple[Optional[ConstitutiveLaw], ...],
        couplings: Tuple[BodyCoupling, ...] = (),
        forces: Tuple[Optional[Force], ...] = (),
        sdf_logics: Optional[Tuple[SDFObjectBase, ...]] = (),
        projection_mass_cutoff: float = 1.0e-10,
        projection_regularization: float = 1.0e-5,
        pressure_stabilization: float = 1.0e-3,
        projection_iterations: int = 32,
        nonnegative_pressure: bool = True,
        **aflip_parameters,
    ):
        if len(couplings) != 1 or couplings[0].skip_mpm_logic:
            raise ValueError(
                "USLIncompressibleAFLIP requires exactly one deformable coupling"
            )
        if len(grid_domains) != 1:
            raise ValueError("USLIncompressibleAFLIP requires exactly one grid domain")
        if not isinstance(constitutive_laws[0], MuI_IC):
            raise TypeError("USLIncompressibleAFLIP currently requires MuI_IC")

        super().__init__(
            grid_domains=grid_domains,
            constitutive_laws=constitutive_laws,
            couplings=couplings,
            forces=forces,
            sdf_logics=sdf_logics,
            **aflip_parameters,
        )
        domain = grid_domains[0]
        if domain.dim not in (2, 3):
            raise ValueError("pressure projection supports only 2D and 3D grids")
        self.cell_grid_size = tuple(
            size if periodic else size - 1
            for size, periodic in zip(domain.grid_size, domain.periodic_axes)
        )
        self.cell_node_hashes, self.cell_node_gradients = self._build_cell_stencil(
            domain
        )
        self.edge_left, self.edge_right = self._build_pressure_edges(domain)
        self.projection_mass_cutoff = projection_mass_cutoff
        self.projection_regularization = projection_regularization
        self.pressure_stabilization = pressure_stabilization
        self.projection_iterations = projection_iterations
        self.nonnegative_pressure = nonnegative_pressure

    @staticmethod
    def _build_cell_stencil(domain):
        """Build compact cell-to-node connectivity for divergence and gradient."""
        dim = domain.dim
        node_grid_size = domain.grid_size
        cell_grid_size = tuple(
            size if periodic else size - 1
            for size, periodic in zip(node_grid_size, domain.periodic_axes)
        )
        transverse_weight = 1.0 / (domain.cell_size * (2 ** (dim - 1)))
        corners = tuple(itertools.product((0, 1), repeat=dim))
        gradients = np.asarray(
            [
                [
                    (-1.0 if corner[axis] == 0 else 1.0) * transverse_weight
                    for axis in range(dim)
                ]
                for corner in corners
            ]
        )
        node_hashes = []
        for cell_idx in itertools.product(*(range(size) for size in cell_grid_size)):
            cell_nodes = []
            for corner in corners:
                node_idx = tuple(
                    (cell_idx[axis] + corner[axis]) % node_grid_size[axis]
                    if domain.periodic_axes[axis]
                    else cell_idx[axis] + corner[axis]
                    for axis in range(dim)
                )
                cell_nodes.append(np.ravel_multi_index(node_idx, node_grid_size))
            node_hashes.append(cell_nodes)
        return jnp.asarray(node_hashes, dtype=jnp.int32), jnp.asarray(gradients)

    @staticmethod
    def _build_pressure_edges(domain):
        """Return unique face-neighbour cell pairs for matrix-free stabilization."""
        cell_grid_size = tuple(
            size if periodic else size - 1
            for size, periodic in zip(domain.grid_size, domain.periodic_axes)
        )
        edges = set()
        for cell_idx in itertools.product(*(range(size) for size in cell_grid_size)):
            cell_hash = int(np.ravel_multi_index(cell_idx, cell_grid_size))
            for axis in range(domain.dim):
                neighbour = list(cell_idx)
                neighbour[axis] += 1
                if neighbour[axis] >= cell_grid_size[axis]:
                    if not domain.periodic_axes[axis]:
                        continue
                    neighbour[axis] = 0
                neighbour_hash = int(
                    np.ravel_multi_index(tuple(neighbour), cell_grid_size)
                )
                if cell_hash != neighbour_hash:
                    edges.add(tuple(sorted((cell_hash, neighbour_hash))))
        sorted_edges = sorted(edges)
        return (
            jnp.asarray([edge[0] for edge in sorted_edges], dtype=jnp.int32),
            jnp.asarray([edge[1] for edge in sorted_edges], dtype=jnp.int32),
        )

    def _get_p2g_stress(self, law, stress_stack):
        if not isinstance(law, MuI_IC):
            return stress_stack
        pressure = jnp.trace(stress_stack, axis1=1, axis2=2) / 3.0
        return stress_stack - pressure[:, None, None] * jnp.eye(3)

    def _get_nodes_sdfs(self, world, sim_cache, dt, time):
        """Use the physical SDF surface for projection and wall traction.

        The generic CPIC path expands nodal SDFs by one grid spacing.  Applying
        that expanded distance to an incompressible pressure boundary moves a
        grid-aligned wall one complete cell into the material.  It also makes
        a pressure traction act on several interior node layers.  Restore the
        physical signed distance here and rebuild the CPIC union mask from it.
        """
        sim_cache, _ = super()._get_nodes_sdfs(world, sim_cache, dt, time)
        grid_union_masks = {}

        for grid_idx, domain in enumerate(self.grid_domains):
            union_mask = jnp.zeros((domain.num_cells,))
            for sdf_idx in range(len(self.sdf_logics)):
                geometry = sim_cache.node_geoms[(grid_idx, sdf_idx)]
                physical_distance = geometry.dists + domain.cell_size
                geometry = eqx.tree_at(
                    lambda value: value.dists,
                    geometry,
                    physical_distance,
                )
                sim_cache.node_geoms[(grid_idx, sdf_idx)] = geometry
                inside_weight = jax.nn.sigmoid(
                    -physical_distance * self.sdf_mp_sharpness
                ).squeeze()
                union_mask = jnp.maximum(union_mask, inside_weight)
            grid_union_masks[grid_idx] = union_mask

        return sim_cache, grid_union_masks

    def _pressure_surface_mask(self, geometry):
        """Select a one-node-thick approximation of the physical SDF surface."""
        half_cell = 0.5 * self.grid_domains[0].cell_size
        return jnp.abs(geometry.dists) <= half_cell

    def _particle_cell_hashes(self, position_stack, domain):
        coordinates = jnp.floor(
            (position_stack - jnp.asarray(domain.origin)) / domain.cell_size
        ).astype(jnp.int32)
        cell_sizes = jnp.asarray(self.cell_grid_size)
        periodic = jnp.asarray(domain.periodic_axes)
        coordinates = jnp.where(periodic, jnp.mod(coordinates, cell_sizes), coordinates)
        coordinates = jnp.clip(coordinates, 0, cell_sizes - 1)
        strides = jnp.asarray(
            [
                _product(self.cell_grid_size[axis + 1 :])
                for axis in range(len(self.cell_grid_size))
            ],
            dtype=jnp.int32,
        )
        return jnp.sum(coordinates * strides, axis=1).astype(jnp.int32)

    def _inverse_density_blocks(self, grid, sim_cache, density):
        dim = grid.dim
        inverse_density = jnp.where(
            grid.mass_stack > self.projection_mass_cutoff,
            1.0 / density,
            0.0,
        )
        blocks = inverse_density[:, None, None] * jnp.eye(dim)[None, :, :]

        # Contact has already imposed the wall velocity.  Do not let pressure
        # correction reintroduce a velocity normal to an SDF boundary.
        for sdf_idx in range(len(self.sdf_logics)):
            geometry = sim_cache.node_geoms[(0, sdf_idx)]
            normal = geometry.normals[:, :dim]
            tangent_projector = jnp.eye(dim) - jnp.einsum("ni,nj->nij", normal, normal)
            blocks = jnp.where(
                (geometry.dists <= 0.0)[:, None, None],
                jnp.einsum(
                    "nij,njk,nkl->nil", tangent_projector, blocks, tangent_projector
                ),
                blocks,
            )
        return blocks

    def _divergence(self, node_vectors):
        """Apply the cell-centered divergence without forming a dense matrix."""
        local_vectors = node_vectors[self.cell_node_hashes]
        return jnp.einsum("acd,cd->a", local_vectors, self.cell_node_gradients)

    def _gradient(self, cell_values, num_nodes):
        """Apply the transpose divergence (weak gradient) by scatter-add."""
        contributions = (
            cell_values[:, None, None] * self.cell_node_gradients[None, :, :]
        )
        return (
            jnp.zeros((num_nodes, self.cell_node_gradients.shape[1]))
            .at[self.cell_node_hashes.reshape(-1)]
            .add(contributions.reshape(-1, self.cell_node_gradients.shape[1]))
        )

    def _projection_operator(self, cell_values, inverse_density_blocks):
        """Apply D R D.T matrix-free, where R contains inverse-density blocks."""
        gradient = self._gradient(cell_values, inverse_density_blocks.shape[0])
        weighted_gradient = jnp.einsum("nde,ne->nd", inverse_density_blocks, gradient)
        return self._divergence(weighted_gradient)

    def _projection_diagonal(self, inverse_density_blocks):
        """Return diag(D R D.T) from the compact cell stencil."""
        local_blocks = inverse_density_blocks[self.cell_node_hashes]
        return jnp.einsum(
            "cd,acde,ce->a",
            self.cell_node_gradients,
            local_blocks,
            self.cell_node_gradients,
        )

    def _graph_laplacian(self, cell_values, active_cells):
        """Apply the active-cell pressure graph Laplacian matrix-free."""
        edge_active = active_cells[self.edge_left] * active_cells[self.edge_right]
        difference = cell_values[self.edge_left] - cell_values[self.edge_right]
        contribution = edge_active * difference
        result = jnp.zeros_like(cell_values)
        result = result.at[self.edge_left].add(contribution)
        return result.at[self.edge_right].add(-contribution)

    def _graph_laplacian_diagonal(self, active_cells):
        """Return the number of active face neighbours for each pressure cell."""
        edge_active = active_cells[self.edge_left] * active_cells[self.edge_right]
        diagonal = jnp.zeros_like(active_cells)
        diagonal = diagonal.at[self.edge_left].add(edge_active)
        return diagonal.at[self.edge_right].add(edge_active)

    def _project_velocity(self, grid, active_cells, inverse_density_blocks, dt):
        velocity = jnp.where(
            (grid.mass_stack > self.projection_mass_cutoff)[:, None],
            grid.moment_nt_stack
            / jnp.where(grid.mass_stack > 0.0, grid.mass_stack, 1.0)[:, None],
            0.0,
        )
        rhs = self._divergence(velocity) / dt

        mask = active_cells.astype(velocity.dtype)
        projection_diagonal = self._projection_diagonal(inverse_density_blocks)
        diagonal_scale = jnp.maximum(jnp.max(mask * projection_diagonal), 1.0)
        gauge_diagonal = (1.0 - mask) + (
            mask * self.projection_regularization * diagonal_scale
        )

        def apply_projection(cell_values):
            masked_values = mask * cell_values
            return (
                mask * self._projection_operator(masked_values, inverse_density_blocks)
                + gauge_diagonal * cell_values
            )

        active_degree = self._graph_laplacian_diagonal(mask)
        stabilization_scale = self.pressure_stabilization * diagonal_scale

        def apply_pressure_system(cell_values):
            return apply_projection(cell_values) + (
                stabilization_scale * self._graph_laplacian(cell_values, mask)
            )

        pressure_diagonal = (
            mask * projection_diagonal
            + gauge_diagonal
            + stabilization_scale * active_degree
        )
        cleanup_diagonal = mask * projection_diagonal + gauge_diagonal
        # ``D.T`` is the weak gradient in the mathematical tension-positive
        # convention, whereas HydraxMPM stores compression-positive pressure.
        # The solved multiplier is therefore minus the physical pressure.
        pressure_multiplier = self._solve_pressure(
            apply_pressure_system, pressure_diagonal, rhs * mask
        )
        if self.nonnegative_pressure:
            pressure_multiplier = jnp.minimum(pressure_multiplier, 0.0)
        pressure = -pressure_multiplier

        pressure_gradient_force = self._gradient(
            pressure_multiplier, grid.mass_stack.shape[0]
        )
        correction = -dt * jnp.einsum(
            "nde,ne->nd", inverse_density_blocks, pressure_gradient_force
        )
        velocity_projected = velocity + correction

        # Pressure-mode stabilization deliberately trades a small divergence
        # residual for a smooth physical pressure.  Remove that residual with
        # the unstabilized projection operator; this cleanup multiplier is not
        # added to the constitutive pressure.
        cleanup_rhs = self._divergence(velocity_projected) / dt * mask
        cleanup_multiplier = self._solve_pressure(
            apply_projection, cleanup_diagonal, cleanup_rhs
        )
        cleanup_gradient = self._gradient(cleanup_multiplier, grid.mass_stack.shape[0])
        velocity_projected = velocity_projected - dt * jnp.einsum(
            "nde,ne->nd", inverse_density_blocks, cleanup_gradient
        )
        divergence_projected = self._divergence(velocity_projected)
        return velocity_projected, pressure, divergence_projected

    def _solve_pressure(self, apply_system, diagonal, rhs):
        """Solve the SPD pressure system with fixed-iteration Jacobi-PCG."""
        diagonal = jnp.maximum(diagonal, 1.0e-20)
        x = jnp.zeros_like(rhs)
        residual = rhs
        z = residual / diagonal
        direction = z
        rz = jnp.dot(residual, z)

        def cg_step(_, values):
            x, residual, direction, rz = values
            system_direction = apply_system(direction)
            denominator = jnp.dot(direction, system_direction)
            active = rz > 1.0e-20
            step = jnp.where(
                active,
                rz / jnp.maximum(denominator, 1.0e-30),
                0.0,
            )
            x_next = x + step * direction
            residual_next = residual - step * system_direction
            z_next = residual_next / diagonal
            rz_next = jnp.dot(residual_next, z_next)
            beta = jnp.where(active, rz_next / jnp.maximum(rz, 1.0e-30), 0.0)
            direction_next = z_next + beta * direction
            return x_next, residual_next, direction_next, rz_next

        x, _, _, _ = jax.lax.fori_loop(
            0,
            self.projection_iterations,
            cg_step,
            (x, residual, direction, rz),
        )
        return x

    def _apply_projected_pressure_friction(
        self, velocity, grid, cell_pressure, active_cells, sim_cache, dt
    ):
        """Add the Coulomb impulse associated with projected wall pressure.

        The ordinary collider sees only the tentative normal-velocity impulse.
        A pressure projection supplies most of the persistent wall reaction, so
        its contribution must also enter the Coulomb friction limit.
        """
        active_pressure = cell_pressure * active_cells
        node_hashes = self.cell_node_hashes.reshape(-1)
        repeated_pressure = jnp.broadcast_to(
            active_pressure[:, None], self.cell_node_hashes.shape
        ).reshape(-1)
        repeated_support = jnp.broadcast_to(
            active_cells[:, None], self.cell_node_hashes.shape
        ).reshape(-1)
        pressure_at_nodes = (
            jnp.zeros_like(grid.mass_stack).at[node_hashes].add(repeated_pressure)
        )
        support_at_nodes = (
            jnp.zeros_like(grid.mass_stack).at[node_hashes].add(repeated_support)
        )
        pressure_at_nodes /= jnp.maximum(support_at_nodes, 1.0)
        boundary_measure = self.grid_domains[0].cell_size ** (grid.dim - 1)
        safe_mass = jnp.where(
            grid.mass_stack > self.projection_mass_cutoff,
            grid.mass_stack,
            1.0,
        )

        for force in self.forces:
            if not isinstance(force, SDFCollider) or 0 not in force.g_idx_list:
                continue
            geometry = sim_cache.node_geoms[(0, force.sdf_idx)]
            normal = geometry.normals[:, : grid.dim]
            wall_velocity = geometry.wall_vels[:, : grid.dim]
            relative_velocity = velocity - wall_velocity
            normal_velocity = jnp.einsum("nd,nd->n", relative_velocity, normal)
            tangential_velocity = relative_velocity - normal_velocity[:, None] * normal
            tangential_speed = jnp.linalg.norm(tangential_velocity, axis=1)
            friction_delta = (
                dt
                * force.base_friction
                * pressure_at_nodes
                * boundary_measure
                / safe_mass
            )
            scale = jnp.maximum(
                0.0,
                1.0 - friction_delta / (tangential_speed + 1.0e-12),
            )
            corrected = (
                wall_velocity
                + normal_velocity[:, None] * normal
                + scale[:, None] * tangential_velocity
            )
            contact_mask = self._pressure_surface_mask(geometry) & (
                grid.mass_stack > self.projection_mass_cutoff
            )
            velocity = jnp.where(contact_mask[:, None], corrected, velocity)
        return velocity

    def _integrate_grid(self, world, mechanics, sim_cache, dt, time):
        world, mechanics, sim_cache = super()._integrate_grid(
            world, mechanics, sim_cache, dt, time
        )
        coupling = self.couplings[0]
        domain = self.grid_domains[0]
        grid = sim_cache.grids[0]
        mp = world.material_points[coupling.p_idx]
        cell_hashes = self._particle_cell_hashes(mp.position_stack, domain)
        active_cells = (
            jnp.zeros((self.cell_node_hashes.shape[0],), dtype=jnp.int32)
            .at[cell_hashes]
            .add(1)
            > 0
        )
        density = jnp.sum(mp.mass_stack) / jnp.sum(mp.volume0_stack)
        inverse_density_blocks = self._inverse_density_blocks(grid, sim_cache, density)
        velocity, cell_pressure, _ = self._project_velocity(
            grid, active_cells, inverse_density_blocks, dt
        )
        velocity = self._apply_projected_pressure_friction(
            velocity, grid, cell_pressure, active_cells, sim_cache, dt
        )
        grid = eqx.tree_at(
            lambda value: value.moment_nt_stack,
            grid,
            velocity * grid.mass_stack[:, None],
        )
        grids = list(sim_cache.grids)
        grids[0] = grid
        sim_cache = eqx.tree_at(lambda value: value.grids, sim_cache, tuple(grids))

        law_states = list(mechanics.constitutive_laws)
        law_state = law_states[coupling.c_idx]
        if not isinstance(law_state, MuIIncompressibleState):
            raise TypeError("incompressible solver received an incompatible law state")
        particle_pressure = cell_pressure[cell_hashes]
        law_states[coupling.c_idx] = eqx.tree_at(
            lambda value: value.pressure_stack,
            law_state,
            particle_pressure,
        )
        mechanics = eqx.tree_at(
            lambda value: value.constitutive_laws,
            mechanics,
            tuple(law_states),
        )
        return world, mechanics, sim_cache

    def _g2p(self, world, mechanics, sim_cache, dt, time):
        world, mechanics, sim_cache = super()._g2p(
            world, mechanics, sim_cache, dt, time
        )
        coupling = self.couplings[0]
        mp_states = list(world.material_points)
        mp = mp_states[coupling.p_idx]
        dim = mp.dim

        trace = jnp.trace(mp.L_stack[:, :dim, :dim], axis1=1, axis2=2) / dim
        L_isochoric = mp.L_stack.at[:, :dim, :dim].add(
            -trace[:, None, None] * jnp.eye(dim)
        )
        determinant = jnp.linalg.det(mp.F_stack[:, :dim, :dim])
        scale = jnp.maximum(determinant, 1.0e-20) ** (-1.0 / dim)
        F_isochoric = mp.F_stack.at[:, :dim, :dim].set(
            mp.F_stack[:, :dim, :dim] * scale[:, None, None]
        )
        mp_states[coupling.p_idx] = eqx.tree_at(
            lambda value: (value.L_stack, value.F_stack, value.volume_stack),
            mp,
            (L_isochoric, F_isochoric, mp.volume0_stack),
        )
        world = eqx.tree_at(
            lambda value: value.material_points, world, tuple(mp_states)
        )
        return world, mechanics, sim_cache

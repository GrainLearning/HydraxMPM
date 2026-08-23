"""Pressure-projected USL-AFLIP solver for incompressible MPM materials."""

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..constitutive_laws.mu_i_rheology import (
    MuI_Incompressible,
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

    divergence_matrix: Float[Array, "num_pressure_cells velocity_dofs"]
    pressure_adjacency: Float[Array, "num_pressure_cells num_pressure_cells"]
    cell_node_incidence: Float[Array, "num_pressure_cells num_nodes"]
    cell_grid_size: Tuple[int, ...] = eqx.field(static=True)
    projection_mass_cutoff: float = eqx.field(static=True)
    projection_regularization: float = eqx.field(static=True)
    pressure_stabilization: float = eqx.field(static=True)
    projection_iterations: int = eqx.field(static=True)
    nonnegative_pressure: bool = eqx.field(static=True)
    wall_ghost_no_slip: bool = eqx.field(static=True)

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
        wall_ghost_no_slip: bool = False,
        **aflip_parameters,
    ):
        if len(couplings) != 1 or couplings[0].skip_mpm_logic:
            raise ValueError(
                "USLIncompressibleAFLIP requires exactly one deformable coupling"
            )
        if len(grid_domains) != 1:
            raise ValueError("USLIncompressibleAFLIP requires exactly one grid domain")
        if not isinstance(constitutive_laws[0], MuI_Incompressible):
            raise TypeError(
                "USLIncompressibleAFLIP currently requires MuI_Incompressible"
            )

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
        self.divergence_matrix = self._build_divergence_matrix(domain)
        divergence_by_node = self.divergence_matrix.reshape(
            (self.divergence_matrix.shape[0], domain.num_cells, domain.dim)
        )
        self.cell_node_incidence = jnp.any(
            jnp.abs(divergence_by_node) > 0.0, axis=2
        ).astype(self.divergence_matrix.dtype)
        self.pressure_adjacency = self._build_pressure_adjacency(domain)
        self.projection_mass_cutoff = projection_mass_cutoff
        self.projection_regularization = projection_regularization
        self.pressure_stabilization = pressure_stabilization
        self.projection_iterations = projection_iterations
        self.nonnegative_pressure = nonnegative_pressure
        self.wall_ghost_no_slip = wall_ghost_no_slip

    @staticmethod
    def _build_divergence_matrix(domain):
        """Build cell-centered divergence of multilinear nodal velocity."""
        dim = domain.dim
        node_grid_size = domain.grid_size
        cell_grid_size = tuple(
            size if periodic else size - 1
            for size, periodic in zip(node_grid_size, domain.periodic_axes)
        )
        num_cells = 1
        for size in cell_grid_size:
            num_cells *= size
        num_nodes = domain.num_cells
        matrix = jnp.zeros((num_cells, num_nodes * dim))
        transverse_weight = 1.0 / (domain.cell_size * (2 ** (dim - 1)))

        # Grid sizes are static Python values, so construction happens once.
        import itertools
        import numpy as np

        for cell_idx in itertools.product(*(range(size) for size in cell_grid_size)):
            cell_hash = int(np.ravel_multi_index(cell_idx, cell_grid_size))
            for corner in itertools.product((0, 1), repeat=dim):
                node_idx = tuple(
                    (cell_idx[axis] + corner[axis]) % node_grid_size[axis]
                    if domain.periodic_axes[axis]
                    else cell_idx[axis] + corner[axis]
                    for axis in range(dim)
                )
                node_hash = int(np.ravel_multi_index(node_idx, node_grid_size))
                for axis in range(dim):
                    sign = -1.0 if corner[axis] == 0 else 1.0
                    dof = node_hash * dim + axis
                    matrix = matrix.at[cell_hash, dof].set(sign * transverse_weight)
        return matrix

    @staticmethod
    def _build_pressure_adjacency(domain):
        """Return face-neighbour adjacency for pressure-mode stabilization."""
        import itertools
        import numpy as np

        cell_grid_size = tuple(
            size if periodic else size - 1
            for size, periodic in zip(domain.grid_size, domain.periodic_axes)
        )
        num_cells = _product(cell_grid_size)
        adjacency = jnp.zeros((num_cells, num_cells))
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
                adjacency = adjacency.at[cell_hash, neighbour_hash].set(1.0)
                adjacency = adjacency.at[neighbour_hash, cell_hash].set(1.0)
        return adjacency

    def _get_p2g_stress(self, law, stress_stack):
        if not isinstance(law, MuI_Incompressible):
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

    def _project_velocity(self, grid, active_cells, inverse_density_blocks, dt):
        dim = grid.dim
        velocity = jnp.where(
            (grid.mass_stack > self.projection_mass_cutoff)[:, None],
            grid.moment_nt_stack
            / jnp.where(grid.mass_stack > 0.0, grid.mass_stack, 1.0)[:, None],
            0.0,
        )
        D = self.divergence_matrix.reshape(
            (self.divergence_matrix.shape[0], grid.mass_stack.shape[0], dim)
        )
        system = jnp.einsum("and,nde,bne->ab", D, inverse_density_blocks, D)
        rhs = jnp.einsum("and,nd->a", D, velocity) / dt

        mask = active_cells.astype(system.dtype)
        system = system * mask[:, None] * mask[None, :]
        diagonal_scale = jnp.maximum(jnp.max(jnp.diag(system)), 1.0)
        inactive_and_gauge_diagonal = jnp.diag(
            (1.0 - mask) + mask * self.projection_regularization * diagonal_scale
        )
        projection_system = system + inactive_and_gauge_diagonal
        active_adjacency = self.pressure_adjacency * mask[:, None] * mask[None, :]
        graph_laplacian = jnp.diag(jnp.sum(active_adjacency, axis=1)) - active_adjacency
        pressure_system = projection_system + (
            self.pressure_stabilization * diagonal_scale * graph_laplacian
        )
        # ``D.T`` is the weak gradient in the mathematical tension-positive
        # convention, whereas HydraxMPM stores compression-positive pressure.
        # The solved multiplier is therefore minus the physical pressure.
        pressure_multiplier = self._solve_pressure(pressure_system, rhs * mask)
        if self.nonnegative_pressure:
            pressure_multiplier = jnp.minimum(pressure_multiplier, 0.0)
        pressure = -pressure_multiplier

        pressure_gradient_force = jnp.einsum("and,a->nd", D, pressure_multiplier)
        correction = -dt * jnp.einsum(
            "nde,ne->nd", inverse_density_blocks, pressure_gradient_force
        )
        velocity_projected = velocity + correction

        # Pressure-mode stabilization deliberately trades a small divergence
        # residual for a smooth physical pressure.  Remove that residual with
        # the unstabilized projection operator; this cleanup multiplier is not
        # added to the constitutive pressure.
        cleanup_rhs = jnp.einsum("and,nd->a", D, velocity_projected) / dt * mask
        cleanup_multiplier = self._solve_pressure(projection_system, cleanup_rhs)
        cleanup_gradient = jnp.einsum("and,a->nd", D, cleanup_multiplier)
        velocity_projected = velocity_projected - dt * jnp.einsum(
            "nde,ne->nd", inverse_density_blocks, cleanup_gradient
        )
        divergence_projected = jnp.einsum("and,nd->a", D, velocity_projected)
        return velocity_projected, pressure, divergence_projected

    def _solve_pressure(self, system, rhs):
        """Solve the SPD pressure system with fixed-iteration Jacobi-PCG."""
        diagonal = jnp.maximum(jnp.diag(system), 1.0e-20)
        x = jnp.zeros_like(rhs)
        residual = rhs - system @ x
        z = residual / diagonal
        direction = z
        rz = jnp.dot(residual, z)

        def cg_step(_, values):
            x, residual, direction, rz = values
            system_direction = system @ direction
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
        weights = self.cell_node_incidence * active_cells[:, None]
        pressure_at_nodes = jnp.einsum("an,a->n", weights, active_pressure)
        pressure_at_nodes = pressure_at_nodes / jnp.maximum(
            jnp.sum(weights, axis=0), 1.0
        )
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

    def _apply_wall_ghost_no_slip(self, velocity, grid, sim_cache):
        """Apply an odd velocity extension across a stationary no-slip wall.

        Quadratic particle-grid interpolation reaches one node through a
        grid-aligned wall. Merely fixing the surface node leaves that ghost
        degree of freedom dynamically inconsistent with the no-slip velocity
        field. Reflecting the nearest fluid-node velocity about the wall value
        preserves the wall location without clamping a finite fluid layer.
        """
        if not self.wall_ghost_no_slip:
            return velocity

        domain = self.grid_domains[0]
        node_position = domain.position_stack
        grid_size = jnp.asarray(domain.grid_size, dtype=jnp.int32)
        periodic_axes = jnp.asarray(domain.periodic_axes)
        strides = jnp.asarray(
            [
                _product(domain.grid_size[axis + 1 :])
                for axis in range(domain.dim)
            ],
            dtype=jnp.int32,
        )

        for force in self.forces:
            if not isinstance(force, SDFCollider) or 0 not in force.g_idx_list:
                continue
            geometry = sim_cache.node_geoms[(0, force.sdf_idx)]
            normal = geometry.normals[:, : domain.dim]
            wall_velocity = geometry.wall_vels[:, : domain.dim]
            reflected_position = (
                node_position - 2.0 * geometry.dists[:, None] * normal
            )
            reflected_index = jnp.rint(
                (reflected_position - jnp.asarray(domain.origin))
                / domain.cell_size
            ).astype(jnp.int32)
            reflected_index = jnp.where(
                periodic_axes,
                jnp.mod(reflected_index, grid_size),
                reflected_index,
            )
            reflected_index = jnp.clip(reflected_index, 0, grid_size - 1)
            reflected_hash = jnp.sum(reflected_index * strides, axis=1)
            ghost_velocity = 2.0 * wall_velocity - velocity[reflected_hash]

            tolerance = 1.0e-6 * domain.cell_size
            surface = jnp.abs(geometry.dists) <= tolerance
            inside = geometry.dists < -tolerance
            has_mass = grid.mass_stack > self.projection_mass_cutoff
            velocity = jnp.where(
                (surface & has_mass)[:, None], wall_velocity, velocity
            )
            velocity = jnp.where(
                (inside & has_mass)[:, None], ghost_velocity, velocity
            )
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
            jnp.zeros((self.divergence_matrix.shape[0],), dtype=jnp.int32)
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
        velocity = self._apply_wall_ghost_no_slip(velocity, grid, sim_cache)
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

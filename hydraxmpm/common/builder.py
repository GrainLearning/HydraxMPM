
import jax

from hydraxmpm.solvers.usl import USLSolver
from ..grid.grid import GridDomain

from ..shapefunctions.mapping import InteractionCache, ShapeFunctionMapping

from ..material_points.material_points import (
    MaterialPointState,
    RigidMaterialPointState,
)

from .simstate import MechanicsState, SimState, WorldState

import equinox as eqx

from ..forces.gravity import Gravity
from ..forces.gridcontact import GridContact

from ..forces.sdf_collider import SDFCollider

from ..forces.damping import Damping

from ..forces.force import BaseForceState, Force

from ..constitutive_laws.constitutive_law import (
    ConstitutiveLawState,
    ConstitutiveLaw,
)
from ..constitutive_laws.linearelastic import LinearElasticLaw
from ..constitutive_laws.newtonfluid import NewtonFluid
from ..constitutive_laws.mu_i_rheology import MuI_LC

from typing import List, Tuple, Optional, Any, Dict, Callable


from ..solvers.coupling import BodyCoupling


from jaxtyping import Array, Float, Int, UInt, Bool

from ..solvers.usl_asflip import USLAFLIP

from ..sdf.sdfcollection import PlaneSDF,CompositeSDF


class Registry(list):
    """A list that returns the index of the item you just added."""

    def add(self, item):
        self.append(item)
        return len(self) - 1


class SimBuilder:
    def __init__(self):

        # world state
        self.particle_states = Registry()
        self.sdf_states = Registry()

        # world logics
        self.sdf_logics = Registry()

        # mechanics states
        self.force_states = Registry()
        self.law_states = Registry()
        self.solver_states = Registry()

        # mechanics logics
        self.grid_domains = Registry()
        self.solver_logics = Registry()
        self.couplings = Registry()
        self.force_logics = Registry()
        self.law_logics = Registry()

    def add_grid(
        self,
        *,
        grid_domain: GridDomain =None,
        origin: tuple | Float[Array, "dim"] = None,
        end: tuple | Float[Array, "dim"] = None,
        cell_size: float | Float[Array, "..."] = None,
        padding: int = 3,
    ) -> int:
        """Defines a physical domain.

        Args:
            origin: The starting coordinates of the grid.
            end: The ending coordinates of the grid.
            cell_size: The size of each cell in the grid.
            padding: Ghost cells. 1 for Linear/Quadratic, 2 for Cubic.
        """
        if grid_domain is None:

            grid_domain = GridDomain.create(origin, end, cell_size, padding=padding)
            
        return self.grid_domains.add(grid_domain)

    def add_material_points(
        self,
        *,
        position_stack,
        is_rigid: Optional[bool] = False,
        **particle_kwargs,
    ) -> int:
        

        if is_rigid:
            rigid_mp_state = RigidMaterialPointState.create(
                position_stack=position_stack,
                **particle_kwargs,
            )
            return self.particle_states.add(rigid_mp_state)

        mp_state = MaterialPointState.create(
            position_stack=position_stack,
            **particle_kwargs,
        )
        return self.particle_states.add(mp_state)

    def add_constitutive_law(
        self,
        *,
        law: ConstitutiveLaw,
        law_state: Optional[ConstitutiveLawState] = None,
        **law_kwargs,
    ) -> int:
        """Adds a Physics Logic object (e.g. LinearElastic)."""

        if isinstance(law, NewtonFluid) and law_state is None:
            law_state = law.create_state_from_density(
                density_stack=law_kwargs.get("density_stack", None)
            )
        elif isinstance(law, MuI_LC) and law_state is None:
            law_state = law.create_state_from_density(
                density_stack=law_kwargs.get("density_stack", None)
            )

        elif isinstance(law, LinearElasticLaw):
            law_state = None
        elif law_state is None:
            raise ValueError("Constitutive law state must be provided for this law.")

        c_id = self.law_logics.add(law)
        self.law_states.add(law_state)
        return c_id

    def add_sdf_object(
        self,
        *,
        sdf_logic,
        center_of_mass=None,
        velocity=None,
        angular_velocity=None,
        rotation=None,
        sdf_state=None,
        return_state=False
    ):
        
        if sdf_state is None:
            # if center of mass is not get it from bounding box center
            if center_of_mass is None:
                bbox_min, bbox_max = sdf_logic.get_world_aabb()
                center_of_mass = 0.5 * (bbox_min + bbox_max)

            sdf_state = sdf_logic.create_state(
                center_of_mass=center_of_mass,
                velocity=velocity,
                angular_velocity=angular_velocity,
                rotation=rotation,
            )

        sdf_idx = self.sdf_logics.add(sdf_logic)

        sdf_state_idx = self.sdf_states.add(sdf_state)

        #  these two must be in sync
        assert sdf_state_idx == sdf_idx, "SDF Logic and State indices out of sync!"
        
        if return_state:
            return sdf_idx, sdf_state
        return sdf_idx

    def couple(
        self,
        *,
        p_idx: int = None,
        g_idx: int = None,
        c_idx: Optional[int] = None,
        sdf_idx: Optional[int] = None,  # future use !
        shapefunction: str = "cubic",
    ) -> int:
        """Adds interaction graph between Material Points and Grid."""
        if p_idx is None:
            p_idx = len(self.particle_states) - 1
        if g_idx is None:
            g_idx = len(self.grid_domains) - 1
        if c_idx is None:
            c_idx = len(self.law_logics) - 1

        mp_state = self.particle_states[p_idx]
        grid_state = self.grid_domains[g_idx]

        skip_mpm_logic = isinstance(mp_state, RigidMaterialPointState)

        couple_idx = len(self.couplings)

        if skip_mpm_logic:
            c_idx = None
            s_idx = None
        else:
            s_idx = couple_idx  # assume one-to-one mapping to solver state

        dim = grid_state.dim
        num_particles = mp_state.position_stack.shape[0]

        shp = ShapeFunctionMapping(
            shapefunction=shapefunction,
            dim=dim,
        )

        couple = BodyCoupling(
            shape_map=shp,
            p_idx=p_idx,
            g_idx=g_idx,
            c_idx=c_idx,
            s_idx=s_idx,  # assume one-to-one mapping to solver state
            skip_mpm_logic=skip_mpm_logic,
        )

        couple_idx = self.couplings.add(couple)  # making sure index is correct

        return couple_idx

    # Forces and force  handles
    def add_gravity(
        self,
        *,
        gravity: Float[Array, "dim"] | tuple,
        g_idx_list: Optional[List[int]] = None,
        p_idx_list: Optional[List[int]] = None,
        is_apply_on_grid: bool = True,
    ) -> int:
        """Adds gravity force to the simulation."""
        g_idx_list = (
            list(range(len(self.grid_domains))) if g_idx_list is None else g_idx_list
        )
        p_idx_list = (
            list(range(len(self.particle_states))) if p_idx_list is None else p_idx_list
        )

        f_idx = len(self.force_logics)
        gravity_force = Gravity(
            is_apply_on_grid=is_apply_on_grid,
            g_idx_list=g_idx_list,
            p_idx_list=p_idx_list,
            f_idx=f_idx,
        )
        f_idx = self.force_logics.add(gravity_force)
        self.force_states.append(gravity_force.create_state(gravity=gravity))

        return f_idx

    def add_damping(
        self,
        *,
        alpha: Float[Array, "dim"] | float,
        g_idx_list: Optional[List[int]] = None,
        p_idx_list: Optional[List[int]] = None,
        is_apply_on_grid: bool = True,
    ) -> int:
        """Adds damping force to the simulation."""
        g_idx_list = (
            list(range(len(self.grid_domains))) if g_idx_list is None else g_idx_list
        )
        p_idx_list = (
            list(range(len(self.particle_states))) if p_idx_list is None else p_idx_list
        )

        f_idx = len(self.force_logics)
        damping_force = Damping(
            is_apply_on_grid=is_apply_on_grid,
            g_idx_list=g_idx_list,
            p_idx_list=p_idx_list,
            f_idx=f_idx,
        )
        f_idx = self.force_logics.add(damping_force)
        self.force_states.append(damping_force.create_state(alpha=alpha))

        return f_idx

    def set_solver(
        self,
        *,
        scheme: str,
        b_idx_list: Optional[List[int]] = None,
        f_idx_list: Optional[List[int]] = None,
        sdf_idx_list: Optional[List[int]] = None,
        **solver_params,
    ):
        """Configures a solver instance linking Particles -> Grid -> Physics."""

        # Set default indices to all if none provided
        b_idx_list = (
            list(range(len(self.couplings))) if b_idx_list is None else b_idx_list
        )
        f_idx_list = (
            list(range(len(self.force_logics))) if f_idx_list is None else f_idx_list
        )

        sdf_idx_list = (
            list(range(len(self.sdf_logics))) if sdf_idx_list is None else sdf_idx_list
        )
        
        couplings = tuple(self.couplings[i] for i in b_idx_list)
        forces = tuple(self.force_logics[i] for i in f_idx_list)
        sdf_logics = tuple(self.sdf_logics[i] for i in sdf_idx_list)


        grid_domains = tuple(self.grid_domains[coupling.g_idx] for coupling in couplings)

        # law can be None if rigid particles are used
        laws = tuple(
            self.law_logics[coupling.c_idx] if coupling.c_idx is not None else None
            for coupling in couplings
        )

        if scheme.lower() == "usl":
            solver = USLSolver(
                constitutive_laws=laws,
                couplings=couplings,
                forces=forces,
                sdf_logics=sdf_logics,
                grid_domains=grid_domains,
                **solver_params,
            )
        elif scheme.lower() == "usl_aflip":
            solver = USLAFLIP(
                constitutive_laws=laws,
                forces=forces,
                couplings=couplings,
                sdf_logics=sdf_logics,
                grid_domains=grid_domains,
                **solver_params,
            )

        # Here we assume that each coupling maps
        # to one solver state
        for coupling in couplings:
            mp_state = self.particle_states[coupling.p_idx]

            if isinstance(mp_state, RigidMaterialPointState):
                solver_state = None
            else:
                solver_state = solver.create_state(mp_state)

            self.solver_states.add(solver_state)

        s_idx = self.solver_logics.add(solver)
        return s_idx

    def build(
        self, time: float = 0.0, step: int = 0, dt: float = 1e-3
    ) -> Tuple[List[eqx.Module] | eqx.Module, SimState]:

        # FUTURE WORK: FSI / Partitioned Solvers
        # If multiple solvers are registered (via calling set_solver multiple times
        # with different b_idx_lists), we currently return a list of solvers.
        # A meta-solver or modified training loop would be required to
        # sequence them (e.g. Fluid Step -> Interaction -> Solid Step).
        # For this thesis, we assume a Monolithic solver (len == 1).

        world_state = WorldState(
            material_points=tuple(self.particle_states),
            sdfs=tuple(self.sdf_states),
        )
        mechanics_state = MechanicsState(
            constitutive_laws=tuple(self.law_states),
            solvers=tuple(self.solver_states),
            forces=tuple(self.force_states),
        )
        sim_state = SimState(
            time=time,
            step=step,
            dt=dt,
            world=world_state,
            mechanics=mechanics_state,
        )

        if len(self.solver_logics) == 1:
            return self.solver_logics[0], sim_state

        return self.solver_logics, sim_state

    def add_body_contact(
        self,
        couple_idx_actor: int = 1,
        couple_idx_receiver: int = 0,
        friction: float = 0.0,
        is_rigid: Optional[bool] = False,
        is_reaction: Optional[bool] = False,
        mass_ratio_limit: Optional[float] = 10.0,
    ):
        """Convenience method for adding a wall."""

        f_idx = len(self.force_logics)

        grid_contact = GridContact(
            couple_idx_actor=couple_idx_actor,
            couple_idx_receiver=couple_idx_receiver,
            friction=friction,
            is_rigid=is_rigid,
            is_reaction=is_reaction,
            mass_ratio_limit=mass_ratio_limit,
        )
        self.force_logics.append(grid_contact)

        self.force_states.append(None)  # Stateless force
        return f_idx

    def add_sdf_collider(
        self,
        sdf_idx: int = None,
        g_idx_list: list[int] = None,
        friction: float = 0.0,
        gap: float = 1e-4,
    ):
        """Convenience method for adding objects."""

        if sdf_idx is None:
            sdf_idx = len(self.sdf_states) - 1

        g_idx_list = (
            list(range(len(self.grid_domains))) if g_idx_list is None else g_idx_list
        )

        sdf_collider = SDFCollider(
            sdf_idx=sdf_idx,
            g_idx_list=g_idx_list,
            gap=gap,
        )

        f_idx = self.force_logics.add(sdf_collider)

        return f_idx

    def summary(self, dt: float = None):
        """Prints an overview of the simulation."""
        print(f"\n{' Simulation Configuration ':=^50}")
        
        # Temporal Information
        if dt is not None:
            print(f"dt = {dt:.2e} s")

        # Grid / Domain Information
        for i, gd in enumerate(self.grid_domains):
            dim = gd.dim
            res = " × ".join(map(str, gd.grid_size))
            bounds = " to ".join([str(gd.origin), str(gd.end)])
            print(f"Grid [{i}]:    {res} nodes | cell size: {gd.cell_size:.4f} | {dim}D Domain")
            print(f"            bounds: {bounds}")

        # Material Points
        total_p = 0
        for i, mp in enumerate(self.particle_states):
            n_p = mp.num_points
            total_p += n_p
            p_type = "Rigid" if isinstance(mp, RigidMaterialPointState) else "Deformable"
            print(f"Body [{i}]:    {n_p} particles ({p_type})")
        print(f"Total |P|:  {total_p}")

        # Physics / Mechanics
        print(f"{' Physics Logics ':-^50}")
        for i, law in enumerate(self.law_logics):
            name = law.__class__.__name__
            print(f"Law [{i}]:     {name}")
            
        for i, force in enumerate(self.force_logics):
            name = force.__class__.__name__
            print(f"Force [{i}]:   {name}")

        # Solvers
        for i, solver in enumerate(self.solver_logics):
            name = solver.__class__.__name__
            print(f"Solver [{i}]:  {name}")
            
        print(f"{'':=^50}\n")
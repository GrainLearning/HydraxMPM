

# import equinox as eqx
# import jax
# import jax.numpy as jnp

# from typing import List, Tuple, Optional

# from hydraxmpm.constitutive_laws.newtonfluid import NewtonFluid

# from ..material_points.material_points import (
#     BaseMaterialPointState,
#     MaterialPointState,
# )
from hydraxmpm.solvers.usl import USLSolver
from ..grid.grid import GridState

from ..shapefunctions.mapping import InteractionCache, ShapeFunctionMapping

# from ..solvers.usl import USLSolver

# from ..solvers.solver import BaseSolver, BaseSolverState

from ..material_points.material_points import (
    MaterialPointState,
    RigidMaterialPointState,
)

from .simstate import SimState

import equinox as eqx

from ..forces.gravity import Gravity
from ..forces.boundary import PlanarBoundaries
from ..forces.gridcontact import GridContact

from ..forces.sdf_collider import SDFCollider


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


class Registry(list):
    """A list that returns the index of the item you just added."""
    def add(self, item):
        self.append(item)
        return len(self) - 1


class SimBuilder:
    def __init__(self):
        self.particle_states = Registry()
        self.grid_states = Registry()
        self.force_states = Registry()
        self.law_states = Registry()
        self.couplings = Registry()
        self.force_logics = Registry()
        self.law_logics = Registry()
        self.solver_logics = Registry()
        self.solver_states = Registry()
        self.intr_caches = {}

    def add_grid(
        self,
        *,
        origin: tuple | Float[Array, "dim"],
        end: tuple | Float[Array, "dim"],
        cell_size: float |  Float[Array, "..."],
        padding: int = 3,
    ) -> int:
        """Defines a physical domain.

        Args:
            origin: The starting coordinates of the grid.
            end: The ending coordinates of the grid.
            cell_size: The size of each cell in the grid.
            padding: Ghost cells. 1 for Linear/Quadratic, 2 for Cubic.
        """

        grid_state = GridState.create(origin, end, cell_size,padding=padding)
        return self.grid_states.add(grid_state)

    def add_material_points(
        self,
        *,
        position_stack,
        is_rigid: Optional[bool] = False,
        **particle_kwargs,
    ) -> int:
        
        if is_rigid:
            jelly_mp_state = RigidMaterialPointState.create(
                position_stack=position_stack,
                **particle_kwargs,
            )
            return self.particle_states.add(jelly_mp_state)

        jelly_mp_state = MaterialPointState.create(
            position_stack=position_stack,
            **particle_kwargs,
        )
        return self.particle_states.add(jelly_mp_state)

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

    def couple(
        self,
        *,
        p_idx: int=None,
        g_idx: int=None,
        c_idx: Optional[int] = None,
        shapefunction: str = "cubic",

    ) -> int:
        """Adds interaction graph between Material Points and Grid."""

        if p_idx is None:
            p_idx = len(self.particle_states) - 1
        if g_idx is None:
            g_idx = len(self.grid_states) - 1
        if c_idx is None:
            c_idx = len(self.law_logics) - 1

        mp_state = self.particle_states[p_idx]
        grid_state = self.grid_states[g_idx]

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
            s_idx=s_idx, # assume one-to-one mapping to solver state
            skip_mpm_logic=skip_mpm_logic,
        )


        intr_cache = shp.create_cache(num_points=num_particles, dim=dim)

        self.intr_caches[(p_idx, g_idx)] = intr_cache
        
        couple_idx = self.couplings.add(couple) # making sure index is correct
    
        return couple_idx

    # Forces and force  handles
    def add_gravity(
        self,
        *,
        gravity: Float[Array, "dim"] | tuple,
        g_idx_list: Optional[List[int]] = None,
        p_idx_list: Optional[List[int]] = None,
        is_apply_on_grid: bool = True
    ) -> int:
        """Adds gravity force to the simulation."""
        g_idx_list = list(range(len(self.grid_states))) if g_idx_list is None else g_idx_list
        p_idx_list = list(range(len(self.particle_states))) if p_idx_list is None else p_idx_list

        f_idx = len(self.force_logics)
        gravity_force = Gravity(
            is_apply_on_grid=is_apply_on_grid,
            g_idx_list=g_idx_list,
            p_idx_list=p_idx_list,
            f_idx=f_idx,
        )
        f_idx =self.force_logics.add(gravity_force)
        self.force_states.append(gravity_force.create_state(gravity=gravity))

        return f_idx

    def add_boundary(
        self,
        *,
        origin: tuple | Float[Array, "dim"],
        end: tuple | Float[Array, "dim"],
        friction: float | Float[Array, "num_walls"] = 0.0,
        g_idx_list: Optional[List[int]] = None,
        gap = 1e-4,
    ) -> int:

        g_idx_list = list(range(len(self.grid_states))) if g_idx_list is None else g_idx_list

        # grid_states = [self.grid_states[g_idx] for g_idx in g_idx_list]
        # max_dim = max(grid.dim for grid in grid_states)
        # min_origin = tuple(
        #     min(grid.origin[d] for grid in grid_states) for d in range(max_dim)
        # )
        # max_end = tuple(
        #     max(grid.end[d] for grid in grid_states) for d in range(max_dim)
        # )

        f_idx = len(self.force_logics)

        boundaries = PlanarBoundaries(
            # origin=min_origin,
            # end=max_end,
            origin=origin,
            end=end,
            frictions=friction,
            f_idx=f_idx,
            g_idx_list=g_idx_list,
            gap=gap
        )
        f_idx = self.force_logics.add(boundaries) # making sure f_idx is correct
        self.force_states.append(None)
    
        return f_idx

    def set_solver(
        self,
        *,
        scheme: str,
        b_idx_list: Optional[List[int]] = None,
        f_idx_list: Optional[List[int]] = None,
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
        
        couplings = tuple(self.couplings[i] for i in b_idx_list)
        forces = tuple(self.force_logics[i] for i in f_idx_list)
        # Law can be None if rigid particles are used
        laws = tuple(
            self.law_logics[coupling.c_idx] if coupling.c_idx is not None
            else None for coupling in couplings)

        if scheme.lower() == "usl":
            solver = USLSolver(
                constitutive_laws=laws,
                couplings =couplings,
                forces=forces,
                **solver_params,
            )
        elif scheme.lower() == "usl_aflip":
            solver = USLAFLIP(
                constitutive_laws=laws,
                forces=forces,
                couplings=couplings,
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
        sim_state = SimState(
            time=time,
            step=step,
            dt=dt,
            material_points=tuple(self.particle_states),
            constitutive_laws=tuple(self.law_states),
            solvers=tuple(self.solver_states),
            interactions=self.intr_caches,
            grids=tuple(self.grid_states),
            forces=tuple(self.force_states),
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
        sdf_object,
        center_of_mass=None,
        velocity=None,
        angular_velocity=None,
        rotation=None,
        f_state=None,
        f_idx: int = None,
        g_idx_list: list[int] = None,
        friction: float = 0.0,
        gap: float = 1e-4,
    ):
        """Convenience method for adding a wall."""

        g_idx_list = list(range(len(self.grid_states))) if g_idx_list is None else g_idx_list

        f_idx = len(self.force_logics)

  
        sdf_collider = SDFCollider(
            sdf_object=sdf_object,
            f_idx=f_idx,
            g_idx_list=g_idx_list,
            friction=friction,
            gap=gap,
        )
        if f_state is not None:
            sdf_collider_state = f_state
        else:
            sdf_collider_state = sdf_collider.create_state(
                center_of_mass=center_of_mass,
                velocity=velocity,
                angular_velocity=angular_velocity,
                rotation=rotation,
            )
        
        f_idx  = self.force_logics.add(sdf_collider)

        self.force_states.append(sdf_collider_state)  # Stateful force
        return f_idx

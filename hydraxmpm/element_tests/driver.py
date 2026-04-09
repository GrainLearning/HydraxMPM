import jax
import jax.numpy as jnp
import equinox as eqx
from ..constitutive_laws.constitutive_law import ConstitutiveLaw


class ElementTestDriver(eqx.Module):
      constitutive_law: ConstitutiveLaw

      is_F_linear_approx: bool = False

      def step(self, mp_state, law_state, L_input, dt, is_F_linear_approx=None):



            if is_F_linear_approx:
                F_inc = jnp.eye(3) - L_input * dt
                F_next = F_inc @ mp_state.F_stack
            else:
                F_next = jax.scipy.linalg.expm(-L_input * dt) @  mp_state.F_stack

            # we temporarily update L_stack on the particle so the law sees it
            mp_temp = eqx.tree_at(
                 lambda m: (m.L_stack, m.F_stack), 
                 mp_state, 
                 (L_input, F_next)
                 )
            


            mp_next, law_next = self.constitutive_law.update(mp_temp, law_state, dt)
            
            # update volume/density
            J = jnp.linalg.det(F_next)
            
            vol_next = J * mp_state.volume0_stack
            
            # mass is kept constant (not used for now...)
            mass_next = mp_state.mass_stack 
            

            # repack
            return eqx.tree_at(
                lambda m: (m.L_stack, m.F_stack, m.stress_stack, m.volume_stack),
                mp_state,
                (mp_next.L_stack, F_next, mp_next.stress_stack, vol_next)
            ), law_next
      

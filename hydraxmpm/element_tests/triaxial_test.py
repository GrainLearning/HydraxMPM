from jax.numpy import jnp
import equinox as eqx
from .driver import ElementTestDriver




class TriaxialTest(eqx.Module):
    solver: ElementTestDriver
    confine: float
    is_undrained: bool = False
    axial_rate: float
    num_steps: int
    dt: float


    def run(self, mp_init, law_init):
        
        def scan_step(carry, _):
            mp, law = carry
            
            # ROOT FINDING (Servo Control) 

            def residual(lateral_rate):
                # Construct L matrix
                L = jnp.zeros((1, 3, 3))
                L = L.at[0,0,0].set(lateral_rate) # L_11 
                L = L.at[0,1,1].set(lateral_rate) # L_22 
                L = L.at[0,2,2].set(self.axial_rate) # L_33
                
                # Trial Step
                mp_out, _ = self.solver.step(mp, law, L, self.dt)
                
                # Error: Current Stress - Target
                # sigma sigma_11
                sigma_xx = mp_out.stress_stack[0, 0, 0]
                return sigma_xx + self.target_pressure

            # Solve
            sol = optx.root_find(
                residual, 
                optx.Newton(rtol=1e-5, atol=1e-5), 
                y0=0.0 # Initial guess for lateral rate
            )
            lat_rate_opt = sol.value
            
            # --- FINAL STEP ---
            # Run the step for real with the optimized rate
            L_final = jnp.zeros((1, 3, 3))
            L_final = L_final.at[0,0,0].set(lat_rate_opt)
            L_final = L_final.at[0,1,1].set(lat_rate_opt)
            L_final = L_final.at[0,2,2].set(self.axial_rate)
            
            mp_next, law_next = self.solver.step(mp, law, L_final, self.dt)
            
            return (mp_next, law_next), mp_next # Carry, Output

        # Run Loop
        _, trajectory = jax.lax.scan(scan_step, (mp_init, law_init), None, length=self.num_steps)
        
        return trajectory
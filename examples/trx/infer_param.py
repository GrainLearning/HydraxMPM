
import matplotlib.pyplot as plt


import hydraxmpm as hdx

import jax.numpy as jnp
import jax


import equinox as eqx

class LoadingConfig(eqx.Module):
    """
    A container for all parameters that might vary between tests.
    """

    # Constant material parameters
    nu: float =0.3
    M:float = 0.9
    lam:float = 0.2
    kap:float = 0.05
    N: float =2.0
    p_ref:float =1000.0
    rho_p:float = 2700.0 # 2.7 * 1000
    OCR:float = 1.0
    confine: float = 50_000.0 # 50 kPa
    
    label_idx:int = 0


class LoadingProcedure(eqx.Module):

    # Constant test settings (do not vary per batch)
    num_steps: int = 2000
    total_time: float = 3600.0 # 1 hour  
    total_strain: float = 0.5 
    is_undrained: bool = False

    def __call__(self, config: LoadingConfig):
        """
        Runs ONE simulation based on the provided config.
        Returns a dictionary or PyTree of results (Trajectories).
        """

        # 1. Update the Constitutive Law with specific params from config
        # We use eqx.tree_at to functionally update the template law

        base_law = hdx.ModifiedCamClay(
            nu=config.nu,
            M=config.M, 
            lam=config.lam,
            kap=config.kap,
            N=config.N,
            p_ref=config.p_ref,
            rho_p=config.rho_p
        )

        law = base_law

        driver = hdx.ElementTestDriver(law)


        p_target = jnp.array([config.confine])
        law_state, stress_stack, density_stack = law.create_state_from_ocr(
            p_stack=p_target, ocr_stack=config.OCR
        )

        # 2. Initialize State (OCR handling)
        # Note: Ensure create_state_from_ocr is JIT-compatible
        p_target = jnp.array([config.confine])
        law_state, stress_stack, density_stack = law.create_state_from_ocr(
            p_stack=p_target, ocr_stack=config.OCR
        )

        mp_state = hdx.MaterialPointState.create(
            stress_stack=stress_stack, density_stack=density_stack
        )

        # 3. Define the Time-Stepping Loop (Scan)
        dt = self.total_time / self.num_steps
        axial_rate = (self.total_strain / self.num_steps) / dt

        # Create the specific test instance for this run
        # (Assuming TriaxialTest is an eqx.Module that holds the solver)
        triaxial_test = hdx.TriaxialTest(
            solver=driver,
            confine=config.confine,
            is_undrained=self.is_undrained,
            axial_rate=jnp.asarray(axial_rate),
            num_steps=self.num_steps,
            dt=jnp.asarray(dt),
            stride=1,
        )

        # Run the simulation
        # The triaxial_test.run should ideally use jax.lax.scan internally
        mp_traj, law_traj = triaxial_test.run(mp_init=mp_state, law_init=law_state)

        # return mp_traj, law_traj, triaxial_test.axial_strain_stack
        return mp_traj.pressure_stack, mp_traj.q_stack
        


TRUE_PARAMS = jnp.array([1.2, 0.086]) # [M, lambda]
# We will calibrate against 3 different confining pressures to be robust
confine_levels = jnp.array([50_000.0, 100_000.0, 200_000.0]) 


truth_configs = jax.vmap(
    lambda p: LoadingConfig(M=TRUE_PARAMS[0], lam=TRUE_PARAMS[1], confine=p)
)(confine_levels)

kernel = LoadingProcedure(
    is_undrained=True,
)

p_targets, q_targets = jax.jit(jax.vmap(kernel))(truth_configs) 


def loss_fn_single_chain(params,  confines, targets):

    M_guess, lam_guess = params # Unpack scalar
    
    
    static_config = LoadingConfig()

    # We broadcast the current M_guess to all 3 confining pressures
    batch_configs = jax.vmap(
        lambda p: eqx.tree_at(
            lambda c: (c.M, c.lam, c.confine), # <--- Target both fields
            static_config,
            (M_guess, lam_guess, p)            # <--- Inject both values
        )
    )(confines)
    
    # Run Simulation
    # Note: We don't need pressure_stack for loss, only q
    _, q_pred = jax.vmap(kernel)(batch_configs)
    
    # MSE Loss (scaled to kPa to keep gradients numerically stable)
    error = (q_pred - targets) / 1000.0
    return jnp.mean(error**2)


print("\n--- Step 2: Initializing Parallel Chains ---")

N_chains = 10  # Number of simultaneous optimization attempts
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)


# Initialize 10 random guesses for M between 0.5 and 2.0
key = jax.random.PRNGKey(42)
M_guesses = jax.random.uniform(k1, (N_chains,), minval=0.5, maxval=1.5)
lam_guesses = jax.random.uniform(k2, (N_chains,), minval=0.04, maxval=0.20)


population = jnp.stack([M_guesses, lam_guesses], axis=1)

print(f"Population Shape: {population.shape}") 

import optax
optimizer = optax.adam(learning_rate=0.02)
opt_state = optimizer.init(population)

@jax.jit
def update_step_2d(pop, opt_state):
    
    # Outer VMAP maps over axis 0 of 'pop' (the chains)
    # So 'loss_fn_2_params' receives a single vector (2,) each time
    per_chain_grad_fn = jax.vmap(
        jax.value_and_grad(loss_fn_single_chain), 
        in_axes=(0,  None, None) 
    )
    

    losses, grads = per_chain_grad_fn(
        pop, confine_levels, q_targets
    )
    
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_pop = optax.apply_updates(pop, updates)
    
    # Constraints: M > 0.1, lam > 0.01
    lower_bounds = jnp.array([0.1, 0.01])
    new_pop = jnp.maximum(new_pop, lower_bounds)
    
    return new_pop, new_opt_state, losses


# -----------------------------------------------------------------------------
# 5. OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
print("\n--- Step 3: Running 2D Optimization ---")

# Store history for plotting trajectory
history_M = []
history_lam = []

for i in range(40):
    population, opt_state, losses = update_step_2d(population, opt_state)
    
    # Save for plotting
    history_M.append(population[:, 0])
    history_lam.append(population[:, 1])

    if i % 5 == 0:
        best_idx = jnp.argmin(losses)
        b_M, b_lam = population[best_idx]
        print(f"Step {i:3d} | Loss: {losses[best_idx]:.5f} | M: {b_M:.4f} | lam: {b_lam:.4f}")


best_idx = jnp.argmin(losses)
best_params = population[best_idx]

print("\n--- Complete ---")
print(f"True Params: M={TRUE_PARAMS[0]}, lam={TRUE_PARAMS[1]}")
print(f"Best Found : M={best_params[0]:.4f}, lam={best_params[1]:.4f}")

# Convert history to arrays for plotting
hist_M = jnp.array(history_M)   # (Steps, N_chains)
hist_lam = jnp.array(history_lam)

plt.figure(figsize=(8, 6))

# Plot the trajectory of EVERY chain
for chain_i in range(N_chains):
    plt.plot(hist_M[:, chain_i], hist_lam[:, chain_i], 'k-', alpha=0.3, linewidth=1)
    # Plot start point
    plt.plot(hist_M[0, chain_i], hist_lam[0, chain_i], 'go', markersize=4)

# Plot the True Target
plt.plot(TRUE_PARAMS[0], TRUE_PARAMS[1], 'r*', markersize=20, label='Ground Truth')

# Plot the Final Best Guess
plt.plot(best_params[0], best_params[1], 'bX', markersize=15, label='Best Estimate')

plt.xlabel('M (Friction)')
plt.ylabel('Lambda (Compressibility)')
plt.title(f'2-Parameter Optimization Trajectories ({N_chains} Chains)')
plt.legend()
plt.grid(True)
plt.show()


# print(p_targets.shape)
# import matplotlib.pyplot as plt

# for i in range(len(confine_levels)):
#     plt.plot(p_targets[i]/1000, q_targets[i]/1000,'-o', label=f'Confine={confine_levels[i]/1000} kPa')
# plt.xlabel('Pressure p (kPa)')
# plt.ylabel('Deviatoric Stress q (kPa)')
# plt.title('Triaxial Compression Ground Truth Paths')

# plt.show()

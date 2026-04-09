import matplotlib.pyplot as plt


# Use CPU for this example
import jax

import hydraxmpm as hdx

import jax.numpy as jnp

import pandas as pd

import equinox as eqx
import optax


import os

base_path = os.path.dirname(os.path.abspath(__file__))

###########################################################################
# Settings
###########################################################################


N_chains = 100
PARAM_SPECS = [
    {"name": "M", "min": 0.5, "max": 2.0},
    {"name": "lam", "min": 0.01, "max": 0.2},
    {"name": "kap_perc", "min": 0.1, "max": 0.9}, 
    {"name": "N", "min": 0.4, "max": 4.0},
    {"name": "nu", "min": 0.1, "max": 0.45},
]

###########################################################################
# Setup default material parameters and loading procedure for single element test
###########################################################################

class LoadingParameters(eqx.Module):
    """
    A container for all material parameters
    """
    nu: float = 0.3
    M: float = 0.9
    lam: float = 0.2
    kap: float = 0.05
    N: float = 2.0
    p_ref: float = 1000.0
    rho_p: float = 2675.0  # 2.7 * 1000
    OCR: float = 1.0
    confine: float = 50_000.0  # 50 kPa
    label_idx: int = 0


class LoadingProcedure(eqx.Module):
    """Procedure to run a single triaxial simulation"""

    num_steps: int = 1000
    axial_rate: float = 0.02
    is_undrained: bool = False
    dt: float = 0.01

    def __call__(self, config: LoadingParameters):
        """
        Runs ONE simulation based on the provided config.

        Returns a dictionary or PyTree of results (Trajectories).
        """


        base_law = hdx.ModifiedCamClay(
            nu=config.nu,
            M=config.M,
            lam=config.lam,
            kap=config.kap,
            N=config.N,
            p_ref=config.p_ref,
            rho_p=config.rho_p,
        )

        law = base_law

        driver = hdx.ElementTestDriver(law)

        p_target = jnp.array([config.confine])

        law_state, stress_stack, density_stack = law.create_state_from_ocr(
            p_stack=p_target, ocr_stack=config.OCR
        )

        p_target = jnp.array([config.confine])
        law_state, stress_stack, density_stack = law.create_state_from_ocr(
            p_stack=p_target, ocr_stack=config.OCR
        )

        mp_state = hdx.MaterialPointState.create(
            stress_stack=stress_stack, density_stack=density_stack
        )


        dt = self.dt


        triaxial_test = hdx.TriaxialTest(
            solver=driver,
            confine=config.confine,
            is_undrained=self.is_undrained,
            axial_rate=jnp.asarray(self.axial_rate),
            num_steps=self.num_steps,
            dt=jnp.asarray(dt),
            stride=1,
        )

        mp_traj, law_traj = triaxial_test.run(mp_init=mp_state, law_init=law_state)

        return mp_traj.pressure_stack/1000, mp_traj.q_stack/1000, triaxial_test.axial_strain_stack*100



kernel = LoadingProcedure(is_undrained=True)

###########################################################################
# Load experimental data
###########################################################################

confine_levels = jnp.array([100_000.0])

df = pd.read_csv(os.path.join(base_path, "data/M02.dat"), sep=r"\s+", header=0, skiprows=[1, 2])



exp_strain = jnp.array(df["eps1"].values)
exp_q = jnp.array(df["q"].values)
exp_p = jnp.array(df["p"].values)

###########################################################################
# Configure upper and lower bounds for parameters, and initialize
###########################################################################


PARAM_NAMES = [p["name"] for p in PARAM_SPECS]
LOWER_BOUNDS = jnp.array([p["min"] for p in PARAM_SPECS])
UPPER_BOUNDS = jnp.array([p["max"] for p in PARAM_SPECS])

num_params = len(PARAM_SPECS)

key = jax.random.PRNGKey(42)

population = jax.random.uniform(key, (N_chains, num_params), 
                                minval=LOWER_BOUNDS, maxval=UPPER_BOUNDS)

###########################################################################
# Define loss function and optimization step
###########################################################################


def loss_fn_single_chain(params, confines):


    # broadcast params to all confine levels in the batch
    batch_configs = jax.vmap(
        lambda p: LoadingParameters(
            confine=p, 
            M=params[0], lam=params[1], kap=params[1]*params[2], N=params[3], nu=params[4]
        )
    )(confines)

    p_pred, q_pred, ax_pred = jax.vmap(kernel)(batch_configs)

    def interpolate_single_test(sim_strain, sim_q):
        return jnp.interp(exp_strain, sim_strain, sim_q)

    q_interp = jax.vmap(interpolate_single_test)(ax_pred, q_pred)


    # MSE Loss (scaled to kPa units)
    error = (q_interp - exp_q) 
    return jnp.mean(error**2)






optimizer = optax.adam(learning_rate=0.02)
opt_state = optimizer.init(population)



###########################################################################
# Update chained optimization loop
###########################################################################

@jax.jit
def update_step_2d(pop, opt_state):


    per_chain_grad_fn = jax.vmap(
        jax.value_and_grad(loss_fn_single_chain), in_axes=(0, None)
    )

    losses, grads = per_chain_grad_fn(pop, confine_levels)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_pop = optax.apply_updates(pop, updates)


    new_pop = jnp.clip(new_pop, LOWER_BOUNDS, UPPER_BOUNDS)

    return new_pop, new_opt_state, losses


###########################################################################
# Run optimization loop
###########################################################################


best_losses = []
history_M = []
history_lam = []

for i in range(101):
    population, opt_state, losses = update_step_2d(population, opt_state)
    
  
    current_best_loss = jnp.min(losses)
    best_losses.append(current_best_loss)


    history_M.append(population[:, 0])
    history_lam.append(population[:, 1])

    if i % 10 == 0:
        best_idx = jnp.argmin(losses)
        best_row = population[best_idx]
        param_str = " | ".join([f"{name}: {val:.4f}" for name, val in zip(PARAM_NAMES, best_row)])
        print(f"Step {i:3d} | Loss: {losses[best_idx]:.5f} | {param_str}")

# Extract final best parameters
best_idx = jnp.argmin(losses)
best_params = population[best_idx]


final_M = best_params[0]
final_lam = best_params[1]
final_kap = final_lam * best_params[2]
final_N = best_params[3]
final_nu = best_params[4]

###########################################################################
# Post-optimization analysis with best parameters
###########################################################################

print(f"M: {final_M:.4f}, lam: {final_lam:.4f}, kap: {final_kap:.4f}, N: {final_N:.4f}, nu: {final_nu:.4f}")

best_config = LoadingParameters(
    confine=confine_levels[0], 
    OCR=1.0, 
    M=final_M, 
    lam=final_lam, 
    kap=final_kap, 
    N=final_N, 
    nu=final_nu
)

p_pred, q_pred, ax_pred = kernel(best_config)


###########################################################################
# Plot axial strain vs q, with experimental data overlay
###########################################################################

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(exp_strain, exp_q, 'ko', markersize=3, alpha=0.5, label="Experimental (M02)")
ax1.plot(ax_pred , q_pred , 'r-', linewidth=2, label="Calibrated MCC Model")
ax1.set_xlabel('axial strain $\\varepsilon_a$ [%]')
ax1.set_ylabel('deviatoric Stress $q$ (kPa)')

ax1.legend()

plt.savefig(os.path.join(base_path, "eps_a_vs_q.png"), dpi=300)

###########################################################################
#   Plot M vs lam example convergence across chains
###########################################################################


fig2, ax2 = plt.subplots(figsize=(7, 5))
hist_M = jnp.array(history_M)
hist_lam = jnp.array(history_lam)


for chain_i in range(0, N_chains, 10): 
    ax2.plot(hist_M[:, chain_i], hist_lam[:, chain_i], 'black', alpha=0.5, linewidth=0.5)


ax2.scatter(hist_M[0, :], hist_lam[0, :], color='green', s=10, alpha=0.5, label='Initial Guesses')
ax2.plot(final_M, final_lam, 'rX', markersize=12, label='Best Estimate')
ax2.set_xlabel('Critical State Friction $M$')
ax2.set_ylabel('Slope of NCL $\lambda$')

ax2.legend()
plt.savefig(os.path.join(base_path, "M_vs_lam.png"), dpi=300)

###########################################################################
#   Plot loss
###########################################################################

fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(best_losses, color='darkblue', linewidth=2)
ax3.set_yscale('log')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('MSE Loss')
ax3.set_title("Calibration Learning Curve")
ax3.grid(True, which="both", ls="-", alpha=0.2)


sensitivity_fn = jax.grad(loss_fn_single_chain)

raw_sensitivities = sensitivity_fn(best_params, confine_levels)

abs_grads = jnp.abs(raw_sensitivities)

plt.savefig(os.path.join(base_path, "loss.png"), dpi=300)

rel_sens = abs_grads * (jnp.abs(best_params) / (best_losses[-1] + 1e-10))
print("\n--- Parameter Sensitivity Analysis ---")
for name, s in zip(PARAM_NAMES, rel_sens):
    print(f"Sensitivity of {name:8s}: {s:.4f}")


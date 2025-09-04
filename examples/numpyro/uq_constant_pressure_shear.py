

import os
from functools import partial
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from all libraries (including JAX)
os.environ["JAX_PLATFORMS"] = "cpu"

import hydraxmpm as hdx
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import equinox as eqx
jax.config.update("jax_enable_x64", True)



from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist


import arviz as az
import matplotlib.pyplot as plt


from jax import random



palette = sns.color_palette("deep")
sns.set_theme(
    context="paper",
    style="white",  # "whitegrid" or "ticks" look modern and clean
    palette="deep",  # Or choose a custom one below
    font_scale=1.5,  # Slightly smaller for tighter layouts
    font="serif",
    rc={
        "lines.linewidth": 3.5,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "xtick.major.pad": 0.0,
        "ytick.major.pad": 0.0,
        "xtick.minor.pad": 0.0,
        "ytick.minor.pad": 0.0,
        "legend.frameon": False,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    },
)


dir_path = os.path.dirname(os.path.realpath(__file__))

plot_dir = os.path.join(dir_path, "./plots")  # plot directory
os.makedirs(plot_dir, exist_ok=True)





N = 3.9918916212749274  # [-] Critical state line slope, from Sorensen et al. (2007)


lam = 0.12501812606291554  # [-]
kap = lam / 3 # [-] bulk modulus

Gs = 2.70   # [-] Specific Gravity  2.65–2.76


v0 =  2.43656481314541

phi_0 = 1 / v0  # [-] initial solid volume fraction

rho_p = Gs*1*1000 #  Particle/ skeleton density 2.7 [g/cm^3] = 2700 [kg/m^3]
rho_0 = phi_0 * rho_p

model = hdx.ModifiedCamClay(
    nu=0.2,
    M= 0.90,
    lam=lam,
    kap=kap,
    d=4e-6,
    OCR=1,
    ln_N=jnp.log(N),
    rho_p=rho_p,
)


p0 = 50_000 # [Pa] Initial pressure, 50 kPa

total_time = 3600  # 1 hour in seconds

prc_total_shear_strain = 25 # [%]

total_shear_strain = prc_total_shear_strain/100

dgamma_dt = total_shear_strain / total_time  # shear rate [1/s]
dt = total_time / 3600  # time step in seconds (per 1 hour)
num_steps = round(total_time / dt)


print(f"{num_steps} steps, dt = {dt}, total_time={total_time}")

applied_dgamma_dt_stack = (
    2 * jnp.ones(num_steps) * dgamma_dt
)

benchmark = hdx.ConstantVolumeShear(
    deps_xy_dt=applied_dgamma_dt_stack,
    num_steps=applied_dgamma_dt_stack.shape[0],
    p0=p0,
    init_material_points=True,  # initialize values of the material point
)

solver = hdx.SIPSolver(
        output_vars=(
            "q_stack",
            "eps_s_stack",
        ),
        constitutive_law=model,
        sip_benchmarks=benchmark,
        is_linear_approx=False,
    )

solver = solver.setup()

(target_q_stack,_eps_s_stack) = solver.run(dt=dt)

plt.plot(_eps_s_stack, target_q_stack/1000, label="obs", color="black")
plt.xlabel("shear strain $\\varepsilon_s$ [%]")
plt.ylabel("deviatoric stress $q$ [kPa]")
plt.title("Constant Volume Shear - Observations")
plt.savefig(f"{plot_dir}/constant_volume_shear_obs.png", dpi=300)

# @jax.jit
def run_mcc(params: dict, solver: hdx.SIPSolver):
    new_solver = eqx.tree_at(
        lambda state: (
            state.constitutive_law.M,
            state.constitutive_law.nu,
        ),
        solver,
        (params["M"], params["nu"]),
    )


    (q_stack,_) = new_solver.run(dt=dt)

    return q_stack


def bayesian_model(solver, q_obs=None):
    # Priors for constitutive parameters
    M = numpyro.sample("M", dist.Uniform(0.7, 1.2))
    nu = numpyro.sample("nu", dist.Uniform(0.1, 0.3))
    noise = numpyro.sample("noise", dist.HalfNormal(0.5))  # Observational noise

    params = dict(M=M, nu=nu)

    pred_q_stack = run_mcc(params, solver)

    numpyro.sample("obs", dist.Normal(pred_q_stack, noise), obs=q_obs)

nuts_kernel = NUTS(bayesian_model,  target_accept_prob=0.9)
mcmc = MCMC(nuts_kernel, num_warmup=40, num_samples=100)
mcmc.run(random.PRNGKey(42), solver, target_q_stack)
mcmc.print_summary()

samples = mcmc.get_samples(group_by_chain=False)
M_pred = samples["M"].mean()
nu_pred = samples["nu"].mean()


# # Convert MCMC data to ArviZ InferenceData
idata = az.from_numpyro(mcmc)
az.plot_posterior(idata, var_names=["M", "nu"],ref_val=[model.M, model.nu])
plt.tight_layout()


plt.savefig(
    plot_dir + "/posterior.png", bbox_inches="tight"
)

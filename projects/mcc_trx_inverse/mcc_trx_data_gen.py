import matplotlib.pyplot as plt


# Use CPU for this example
import jax

import hydraxmpm as hdx

import jax.numpy as jnp
import numpy as np
import h5py
from scipy.stats import qmc

import pandas as pd

import equinox as eqx


import os

base_path = os.path.dirname(os.path.abspath(__file__))

###########################################################################
# Settings
###########################################################################


N_samples = 1000

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
    is_p_constant: bool = False
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

        mp_state = hdx.MaterialPointState.create(
            stress_stack=stress_stack, density_stack=density_stack
        )


        dt = self.dt


        triaxial_test = hdx.TriaxialTest(
            solver=driver,
            confine=config.confine,
            is_undrained=self.is_undrained,
            is_p_constant=self.is_p_constant,
            axial_rate=jnp.asarray(self.axial_rate),
            num_steps=self.num_steps,
            dt=jnp.asarray(dt),
            stride=1,
        )

        mp_traj, law_traj = triaxial_test.run(mp_init=mp_state, law_init=law_state)

        return (
            triaxial_test.axial_strain_stack * 100,
            mp_traj.stress_stack / 1000,
            mp_traj.eps_stack,
        )

DATASET_MODES = {
    "drained": {"is_undrained": False, "is_p_constant": True},
    "undrained": {"is_undrained": True, "is_p_constant": False},
}

###########################################################################
# Configure bounds and Latin hypercube sampling
###########################################################################

confine_levels = jnp.array([100_000.0])
PARAM_NAMES = [p["name"] for p in PARAM_SPECS]
LOWER_BOUNDS = jnp.array([p["min"] for p in PARAM_SPECS])
UPPER_BOUNDS = jnp.array([p["max"] for p in PARAM_SPECS])

num_params = len(PARAM_SPECS)


def run_single_sample(params, kernel):
    batch_configs = jax.vmap(
        lambda p: LoadingParameters(
            confine=p,
            M=params[0],
            lam=params[1],
            kap=params[1] * params[2],
            N=params[3],
            nu=params[4],
        )
    )(confine_levels)

    ax_pred, stress_pred, strain_pred = jax.vmap(kernel)(batch_configs)
    return ax_pred[0], stress_pred[0], strain_pred[0]


rng = np.random.default_rng(42)
sampler = qmc.LatinHypercube(d=num_params, seed=rng)
lhs_unit = sampler.random(n=N_samples)
population = jnp.asarray(
    qmc.scale(lhs_unit, np.asarray(LOWER_BOUNDS), np.asarray(UPPER_BOUNDS))
)

sample_ids = np.arange(N_samples)
param_np = np.asarray(population)

df_params = pd.DataFrame(param_np, columns=PARAM_NAMES)
df_params.insert(0, "sample_id", sample_ids)
df_params["kap"] = df_params["lam"] * df_params["kap_perc"]

for mode_name, mode_flags in DATASET_MODES.items():
    is_undrained = mode_flags["is_undrained"]
    is_p_constant = mode_flags["is_p_constant"]
    kernel = LoadingProcedure(
        is_undrained=is_undrained,
        is_p_constant=is_p_constant,
    )
    ax_all, stress_all, strain_all = jax.vmap(
        lambda params: run_single_sample(params, kernel)
    )(population)

    ax_np = np.asarray(ax_all)
    stress_np = np.asarray(stress_all)
    strain_np = np.asarray(strain_all)
    n_steps = ax_np.shape[1]

    ###########################################################################
    # Save generated dataset
    ###########################################################################

    h5_path = os.path.join(base_path, f"lhs_dataset_{mode_name}.h5")
    with h5py.File(h5_path, "w") as h5f:
        h5f.attrs["n_samples"] = N_samples
        h5f.attrs["n_steps"] = n_steps
        h5f.attrs["param_names"] = np.array(PARAM_NAMES, dtype="S")
        h5f.attrs["mode"] = mode_name
        h5f.attrs["is_undrained"] = int(is_undrained)
        h5f.attrs["is_p_constant"] = int(is_p_constant)

        samples_group = h5f.create_group("samples")

        for i in range(N_samples):
            sample_group = samples_group.create_group(f"{i:04d}")
            sample_group.create_dataset("sample_id", data=np.int32(i))

            params_group = sample_group.create_group("parameters")
            params_group.create_dataset("M", data=param_np[i, 0])
            params_group.create_dataset("lam", data=param_np[i, 1])
            params_group.create_dataset("kap_perc", data=param_np[i, 2])
            params_group.create_dataset("kap", data=param_np[i, 1] * param_np[i, 2])
            params_group.create_dataset("N", data=param_np[i, 3])
            params_group.create_dataset("nu", data=param_np[i, 4])

            seq_group = sample_group.create_group("sequences")
            seq_group.create_dataset("axial_strain_pct", data=ax_np[i])
            seq_group.create_dataset("stress_tensor_kpa", data=stress_np[i])
            seq_group.create_dataset("strain_tensor", data=strain_np[i])

    ###########################################################################
    # Quick visual checks
    ###########################################################################

    fig1, (ax1, ax1b) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Compute scalar p and q only when needed for plotting
    p_plot = np.asarray(hdx.get_pressure(stress_all))
    q_plot = np.asarray(hdx.get_q_vm(stress_all))
    eps_v_plot = np.trace(strain_np, axis1=-2, axis2=-1) * 100.0

    for i in range(min(20, N_samples)):
        ax1.plot(ax_np[i], q_plot[i] / p_plot[i], alpha=0.4, linewidth=1.0)
        if is_undrained:
            ax1b.plot(ax_np[i], p_plot[i], alpha=0.4, linewidth=1.0)
        else:
            ax1b.plot(ax_np[i], eps_v_plot[i], alpha=0.4, linewidth=1.0)

    ax1.set_xlabel("axial strain $\\varepsilon_a$ [%]")
    ax1.set_ylabel("stress ratio $q/p$ (-)")
    ax1.set_title(f"$q/p$ evolution ({mode_name}, first 20)")

    ax1b.set_xlabel("axial strain $\\varepsilon_a$ [%]")
    if is_undrained:
        ax1b.set_ylabel("pressure $p$ (kPa)")
        ax1b.set_title(f"$p$ evolution ({mode_name}, first 20)")
    else:
        ax1b.set_ylabel("volumetric strain $\\varepsilon_v$ [%]")
        ax1b.set_title(f"$\\varepsilon_v$ evolution ({mode_name}, first 20)")

    fig1.tight_layout()
    plt.savefig(os.path.join(base_path, f"lhs_p_q_curves_{mode_name}.png"), dpi=300)

    print(f"Generated {N_samples} LHS samples ({mode_name})")
    print(f"Saved hierarchical dataset: {h5_path}")

    # Read the hdf5 file and print the tensor shapes for the first sample
    with h5py.File(h5_path, "r") as h5f:
        n_samples = h5f.attrs["n_samples"]
        n_steps = h5f.attrs["n_steps"]
        param_names = h5f.attrs["param_names"]

        print(f"Number of samples: {n_samples}")
        print(f"Number of steps per sample: {n_steps}")
        print(f"Parameter names: {param_names}")

        first_sample_group = h5f["samples"]["0000"]
        axial_strain = first_sample_group["sequences"]["axial_strain_pct"][:]
        stress_tensor = first_sample_group["sequences"]["stress_tensor_kpa"][:]
        strain_tensor = first_sample_group["sequences"]["strain_tensor"][:]

        # Compute p and q from stored stress tensor only when needed
        p_kpa = np.asarray(hdx.get_pressure(jnp.asarray(stress_tensor)))
        q_kpa = np.asarray(hdx.get_q_vm(jnp.asarray(stress_tensor)))

        print(f"Axial strain shape: {axial_strain.shape}")
        print(f"Pressure shape: {p_kpa.shape}")
        print(f"Deviatoric stress shape: {q_kpa.shape}")
        print(f"Stress tensor shape: {stress_tensor.shape}")
        print(f"Strain tensor shape: {strain_tensor.shape}")

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.scatter(df_params["M"], df_params["lam"], s=18, alpha=0.7)
ax2.set_xlabel("Critical state friction $M$")
ax2.set_ylabel("Slope of NCL $\\lambda$")
ax2.set_title("LHS samples in parameter space")
plt.savefig(os.path.join(base_path, "lhs_M_vs_lam.png"), dpi=300)
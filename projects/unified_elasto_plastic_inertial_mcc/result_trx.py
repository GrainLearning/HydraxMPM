import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

import matplotlib.pyplot as plt

from matplotlib import colormaps
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import equinox as eqx
import numpy as np
from enum import Enum


import hydraxmpm as hdx
from plotting import TriaxialDashboard, FIGURE_DIR, FIG_SIZES

from sip_setup import SIPConfig, NumericalProcedure


# ###########################################################################
# Simulation setup
# ###########################################################################

class Mode(Enum):
    VARY_SHEAR_RATE = 1
    VARY_PARAM = 2
    VARY_OCR = 3

mode = Mode.VARY_SHEAR_RATE

loading_procedure = NumericalProcedure(is_undrained=False)

ref_config = SIPConfig()


# overconsolidation ratio for varying OCR case
OCR = 2.5

# speeds
hour_in_seconds = 3600.0
slow = 28 * 24 * hour_in_seconds  # 4 weeks
intermediate = 2 * 24 * hour_in_seconds  # 2 days
fast = 24 * hour_in_seconds  # 1 day

kPa = 1000.0

# ###########################################################################
# Setup config
# ###########################################################################


if mode == Mode.VARY_SHEAR_RATE:
    configs_list = [
        SIPConfig(total_time=slow, label_idx=0),  # slow
        SIPConfig(total_time=intermediate, label_idx=1),  # intermediate
        SIPConfig(total_time=fast, label_idx=2),  # fast
    ]
elif mode == Mode.VARY_PARAM:
    configs_list = [
        # rate independent reference
        SIPConfig(total_time=fast, I_v=1e20, I_M=1e20, label_idx=0),
        # M_I (up)
        SIPConfig(total_time=fast, I_v=1e20, label_idx=1),  # v_I (up)
        SIPConfig(total_time=fast, I_M=1e20, label_idx=2),
        SIPConfig(total_time=fast, label_idx=3),  # v_I (up) M_I (up)
    ]
elif mode == Mode.VARY_OCR:
    configs_list = [
        SIPConfig(
            total_time=slow, confine=ref_config.confine / OCR, OCR=OCR, label_idx=0
        ),
        SIPConfig(
            total_time=fast, confine=ref_config.confine / OCR, OCR=OCR, label_idx=1
        ),
    ]

if mode == Mode.VARY_SHEAR_RATE:
    color_maps = plt.cm.plasma(np.linspace(0.3, 0.7, 3))
    mode_label = "vary_shear_rate"
    config_labels = {
        0:lambda eps_a: (
            r"$\dot{\varepsilon}_{a,1} = " + f"{eps_a:.2f}" + r"~\%/\mathrm{h}$",
            0,
        ),  # slow
        1:lambda eps_a: (
            r"$\dot{\varepsilon}_{a,2} = " + f"{eps_a:.2f}" + r"~\%/\mathrm{h}$",
            0,
        ),  # intermediate
        2:lambda eps_a: (
            r"$\dot{\varepsilon}_{a,3} = " + f"{eps_a:.2f}" + r"~\%/\mathrm{h}$",
            0,
        ),  # fast
    }
    I_YS_labels = {
        0:lambda eps_a: (r"I-YS $(\dot{\varepsilon}_{a,1}$)", 0),  # fast
        1:lambda eps_a: (r"I-YS $(\dot{\varepsilon}_{a,2}$)", 0),  # intermediate
        2:lambda eps_a: (r"I-YS $(\dot{\varepsilon}_{a,3}$)", 0),  # slow
    }
    I_SS_labels = {
        0:lambda eps_a: (r"ISS $(\dot{\varepsilon}_{a,1}$)", 0),  # fast
        1:lambda eps_a: (r"ISS $(\dot{\varepsilon}_{a,2}$)", 0),  # intermediate
        2:lambda eps_a: (r"ISS $(\dot{\varepsilon}_{a,3}$)", 0),  # slow
    }
elif mode == Mode.VARY_PARAM:
    color_maps = colormaps.get_cmap("Dark2")(range(len(configs_list)))
    config_labels = {
        0:lambda _: (r"C1: $M_I = v_I =1$", 0),
        1: lambda _: (r"C2: $M_I \uparrow$", 0),
        2: lambda _: (r"C3: $v_I \uparrow$", 0),
        3: lambda _: (r"C4: $M_I \uparrow$, $v_I \uparrow$", 0),
    }
    I_YS_labels = {
        0:lambda _: (r"I-YS (C1)", 0),
        1:lambda _: (r"I-YS (C2)", 0),
        2:lambda _: (r"I-YS (C3)", 0),
        3:lambda _: (r"I-YS (C4)", 0),
    }
    I_SS_labels = {
        0:lambda _: (r"ISS (C1)", 0),
        1:lambda _: (r"ISS (C2)", 0),
        2:lambda _: (r"ISS (C3)", 0),
        3:lambda _: (r"ISS (C4)", 0),
    }
    mode_label = "vary_param"
elif mode == Mode.VARY_OCR:
    color_maps = plt.cm.plasma(np.linspace(0.3, 0.7, 2))
    mode_label = "vary_OCR"
    config_labels = {
        0:lambda _: (r"OCR=2.5, $\dot{\varepsilon}_{a,1}$", 1),
        1:lambda _: (r"OCR=2.5, $\dot{\varepsilon}_{a,3}$", 0),
    }
    I_YS_labels = {
        0: lambda _: (r"I-YS ($\dot{\varepsilon}_{a,1}$)", 0),  # fast
        1: lambda _: (r"I-YS ($\dot{\varepsilon}_{a,3}$)", 0),  # intermediate
    }
    I_SS_labels = {
        0: lambda _: (r"ISS ($\dot{\varepsilon}_{a,1}$)", 0),  # fast
        1: lambda _: (r"ISS ($\dot{\varepsilon}_{a,3}$)", 0),  # intermediate
    }

if loading_procedure.is_undrained:
    p_lim = (0, 120)
    q_lim = (-1.5, 65)
    eps_lim = (-1, 51)
    eps_v_lim = None

    if mode == Mode.VARY_OCR:
        p_lim = (0, 115)
        v_lim = (2.3, 2.4)
        v_ticks = [2.25, 2.30, 2.35, 2.40]
        p_log_lim, p_log_ticks = (15, 50), [10, 20, 30, 40, 50, 60]
    else:
        v_lim = (2.2, 2.3)
        v_ticks = [2.22, 2.24, 2.26, 2.28, 2.30]
        p_log_lim, p_log_ticks = (30, 60), [30, 40, 50, 60]

else:
    eps_lim = (-1, 52)
    eps_v_lim = (0,7.5)

    
    if mode == Mode.VARY_OCR:
        p_lim = (0, 105)
        q_lim = (-1.5, 60)
        v_lim = (2.25, 2.4)
        v_ticks = [2.25, 2.30, 2.35, 2.40]
        p_log_lim, p_log_ticks = (15, 50), [10, 20, 30, 40, 50]
    else:
        p_lim = (0, 180)
        q_lim = (0, 100)
        v_lim = (2.05, 2.35)
        v_ticks = [2.05, 2.10, 2.15, 2.20, 2.25, 2.30]
        p_log_lim, p_log_ticks = (45, 100), [50, 60, 70, 80, 90, 100]



if mode == Mode.VARY_PARAM:
    figsize_key = "full_page_2_extended"
else:
    figsize_key = "full_page_2"

# ###########################################################################
# Run Simulations
# ###########################################################################

batch_config = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *configs_list)

run_batch = jax.vmap(loading_procedure)

jitted_run_batch = jax.jit(run_batch)

batch_mp_traj, batch_law_traj, batch_strain, batch_axial_rate = jitted_run_batch(batch_config)

print("Simulation Complete.")

# ###########################################################################
# Get plotting limits
# ###########################################################################



dashboard = TriaxialDashboard(
    is_undrained=loading_procedure.is_undrained, figsize_key=figsize_key
)

undrained_label = "cu" if loading_procedure.is_undrained else "cd"


dashboard.configure_limits(
    p_lim=p_lim,
    q_lim=q_lim,
    eps_lim=eps_lim,
    v_lim=v_lim,
    p_log_lim=p_log_lim,
    v_ticks=v_ticks,
    p_log_ticks=p_log_ticks,
    eps_v_lim = eps_v_lim,
)

for i, sim_config in enumerate(configs_list):
    print(
        f"Plotting simulation {i+1} of {len(configs_list)} undrained={loading_procedure.is_undrained}"
    )

    p = batch_mp_traj.pressure_stack[i] / kPa
    q = batch_mp_traj.q_stack[i] / kPa
    rho = batch_mp_traj.density_stack[i]
    eps_a = batch_strain[i] * 100
    eps_v = batch_mp_traj.eps_v_stack[i] * 100

    # deviatoric strain rate and shear strain (converted to %) for plotting
    shear_strain_rate = batch_mp_traj.shear_strain_rate_stack[i]

    shear_strain = batch_mp_traj.shear_strain_stack[i] * 100

    v = sim_config.rho_p / rho

    I = (shear_strain_rate * sim_config.d) / jnp.sqrt(p * kPa / sim_config.rho_p)

    # note at steady state deviatoric shear strain is entirely plastic
    I_p = I[-1]

    label, zorder = config_labels[sim_config.label_idx](batch_axial_rate[i])
    i_ys_label, _ = I_YS_labels[sim_config.label_idx](batch_axial_rate[i])
    i_ss_label, _ = I_SS_labels[sim_config.label_idx](batch_axial_rate[i])
    color = color_maps[sim_config.label_idx]

    dashboard.plot_trajectory(
        p=p,
        q=q,
        v=v,
        I_p=I_p,
        eps_s=shear_strain,
        eps_v=eps_v,
        label=label,
        color=color,
        zorder=zorder,
    )
    dashboard.add_trajectory_reference(
        sim_config=sim_config,
        p=p,
        q=q,
        v=v,
        I_p=I_p,
        color=color,
        i_ys_label=i_ys_label,
        i_ss_label=i_ss_label,
    )

dashboard.add_reference(sim_config=sim_config)

output_dir = FIGURE_DIR / f"results_{mode_label}_{undrained_label}"
bottom_space = 0.30
ncol = 4
if mode == Mode.VARY_OCR:
    bottom_space = 0.26
    ncol = 5
elif mode == Mode.VARY_PARAM:
    bottom_space = 0.32
    ncol = 4

dashboard.finalize(
    output_dir,
    ncol=ncol,
    hide_legend=loading_procedure.is_undrained,
    bottom_space=bottom_space,
)

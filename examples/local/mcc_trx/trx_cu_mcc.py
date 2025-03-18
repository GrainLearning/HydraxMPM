import os

# element tests run faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

import hydraxmpm as hdx

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import matplotlib as mpl

import scienceplots

plt.style.use(["science", "no-latex"])


import equinox as eqx
import os

jax.config.update("jax_enable_x64", True)


dir_path = os.path.dirname(os.path.realpath(__file__))


mpl.rcParams["lines.linewidth"] = 1.5

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# initial pressure
p_0 = 150000.0

model = hdx.ModifiedCamClay(
    nu=0.2,
    M=1.2,
    lam=0.025,
    kap=0.005,
    ln_N=0.7,
    d=0.005,
    rho_p=2450,
    R=8,  # Change OCR here
    other=dict(label="MCC OCR=1", ls="-", zorder=-1, color=cycle[0]),
)

et_benchmark = hdx.TRX_CU(
    name="constant pressure slow",
    deps_zz_dt=2.0,
    p0=p_0,
    init_material_points=True,
)


solver = hdx.ETSolver(
    material_points=hdx.MaterialPoints(
        p_stack=p_0,
    ),
    config=hdx.Config(
        num_steps=40000,
        dt=0.00001,
        output=(
            "p_stack",
            "q_stack",
            "gamma_stack",
            "eps_v_stack",
            "specific_volume_stack",
        ),
        file=__file__,
    ),
    constitutive_law=model,
    et_benchmarks=et_benchmark,
)

solver = solver.setup()

(
    p_stack,
    q_stack,
    gamma_stack,
    eps_v_stack,
    specific_volume_stack,
) = solver.run_jit()


fig_resp, axes_resp = plt.subplots(
    ncols=3,
    nrows=1,
    figsize=(8, 2),
    dpi=300,
    layout="constrained",
)

# These are helper functions to plot the results

hdx.make_plot(
    axes_resp.flat[0],
    gamma_stack,
    q_stack,
    color=model.other["color"],
    xlabel="$\\gamma$ [-]",
    ylabel="$q$ [Pa]",
    linestyle="-",
    xlim=(-0.01, None),
)


hdx.make_plot(
    axes_resp.flat[1],
    p_stack,
    q_stack,
    color=model.other["color"],
    xlabel="$p$ [Pa]",
    ylabel="$q$ [Pa]",
    linestyle="-",
    xlim=(0, None),
    # ylim=(0, None),
)


hdx.make_plot(
    axes_resp.flat[2],
    p_stack,
    specific_volume_stack,
    color=model.other["color"],
    xlabel="$p$ (log10-scale) [Pa]",
    ylabel="specific volume (log10-scale) [Pa]",
    linestyle="-",
    xlogscale=True,
    ylogscale=True,
)

# plot additional information
# CSL, ICL, OCL/SL
p_q_p = jnp.arange(0, 400000, 10)
q_csl = model.CSL_q_p(p_q_p)
axes_resp.flat[1].plot(p_q_p, q_csl, "r-", lw=1.0, zorder=-1)
axes_resp.flat[1].set_xlim(0, 400000)
axes_resp.flat[1].set_xlim(0, 400000)


# # ln v - ln p
p_lnv_lnp = jnp.arange(500, 1000000, 10)
p_sl = jnp.arange(500, model.R * 1000000, 10)
v_icl = model.ICL(p_lnv_lnp)
axes_resp.flat[2].plot(
    p_lnv_lnp,
    v_icl,
    color="black",
    ls="-",
    lw=1.0,
    label="ICL",
    zorder=-1,
)

v_csl = model.CSL(p_lnv_lnp)
axes_resp.flat[2].plot(
    p_lnv_lnp, v_csl, color="red", ls="-", lw=1.0, zorder=-1, label="CSL"
)
axes_resp.flat[2].set_xticks([10000, 100000, 1000000], minor=True)
axes_resp.flat[2].set_xlim(10000, 1000000)
axes_resp.flat[2].set_ylim(1.4, 1.65)

fig_resp.suptitle(f"TRX CU for MCC OCR = {int(model.R)}")
plt.savefig(f"{dir_path}/trx_mcc.png")

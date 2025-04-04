import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from all libraries (including JAX)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

import hydraxmpm as hdx

import matplotlib.pyplot as plt


import scienceplots


import os

plt.style.use(["science", "no-latex"])

dir_path = os.path.dirname(os.path.realpath(__file__))

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


p_0 = 150000.0

rho_p = 2450


models = (
    hdx.ModifiedCamClay(
        nu=0.2,
        M=1.2,
        lam=0.025,
        kap=0.005,
        ln_N=0.7,
        d=0.005,
        rho_p=rho_p,
        R=1,
        other=dict(label="MCC OCR=1", ls="-", zorder=-1, color=cycle[0]),
    ),
    hdx.ModifiedCamClay(
        nu=0.2,
        M=1.2,
        lam=0.025,
        kap=0.005,
        ln_N=0.7,
        d=0.005,
        rho_p=rho_p,
        R=3,
        other=dict(label="MCC OCR=3", ls="--", zorder=-1, color=cycle[0]),
    ),
)


sip_benchmarks = (
    hdx.TRX_CU(
        deps_zz_dt=4.0,
        p0=p_0,
        num_steps=4000,
        init_material_points=True,
        other=dict(type="TRX_CU"),
    ),
    hdx.TRX_CD(
        deps_zz_dt=4.0,
        p0=p_0,
        num_steps=4000,
        init_material_points=True,
        other=dict(type="TRX_CD"),
    ),
)


fig, ax = plt.subplots(
    figsize=(4, 3),
    dpi=300,
    layout="constrained",
)
for model in models:
    for sip_benchmark in sip_benchmarks:
        solver = hdx.SIPSolver(
            material_points=hdx.MaterialPoints(
                p_stack=jnp.array([p_0]),
            ),
            output_dict=(
                "p_stack",
                "q_stack",
                "specific_volume_stack",
            ),
            constitutive_law=model,
            sip_benchmarks=sip_benchmark,
        )

        solver = solver.setup()

        (
            p_stack,
            q_stack,
            specific_volume_stack,
        ) = solver.run(0.0001)

        t_stack = jnp.arange(0, sip_benchmark.num_steps) * 0.0001

        hdx.make_plot(
            ax,
            p_stack,
            q_stack,
            xlim=(0, 500000),
            ylim=(0, 500000),
            color=model.other["color"],
            linestyle=model.other["ls"],
        )


hdx.make_plot(
    ax,
    (0, 600000),
    (0, 600000 * models[0].M),
    color="red",
    xlim=(0, 500000),
    ylim=(0, 500000),
    start_end_markers=False,
)

plt.show()

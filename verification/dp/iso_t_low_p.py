from cProfile import label
import os
from turtle import color

from matplotlib import lines


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from all libraries (including JAX)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

import hydraxmpm as hdx

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt


import scienceplots

import sys
from pathlib import Path

# Add the `utils` directory to the path
utils_path = Path(__file__).parent.parent / "utils/"

print(utils_path)
sys.path.append(str(utils_path))

from plots import plot_q_vs_p, make_subplots, plot_q_vs_p_M

import equinox as eqx
import os

plt.style.use(["science", "no-latex"])

dir_path = os.path.dirname(os.path.realpath(__file__))

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


p_0 = 10.0

rho_0 = 2650
K = 7e5  # [Pa]
E = 3 * K * (1 - 2 * 0.3)
models = (
    hdx.DruckerPrager(
        nu=0.3,
        E=E,
        mu_1=0.7,
        rho_0=rho_0,
        other=dict(label="DP P.P", ls="-", zorder=-1, color=cycle[0]),
    ),
)


et_benchmarks = (
    # hdx.TRX_CU(
    #     deps_zz_dt=4.0,
    #     p0=p_0,
    #     init_material_points=True,
    #     other=dict(type="TRX_CU"),
    # ),
    # hdx.TRX_CD(
    #     deps_zz_dt=4.0,
    #     p0=p_0,
    #     init_material_points=True,
    #     other=dict(type="TRX_CD"),
    # ),
    hdx.ISO_C(  # tension
        deps_xx_yy_zz_dt=-1.0,
        p0=p_0,
        init_material_points=True,
        other=dict(type="ISO_C"),
    ),
)


fig, ax = make_subplots()
for model in models:
    for et_benchmark in et_benchmarks:
        solver = hdx.ETSolver(
            material_points=hdx.MaterialPoints(
                p_stack=jnp.array([p_0]),
            ),
            config=hdx.Config(
                num_steps=1000,
                dt=0.0001,
                output=(
                    "p_stack",
                    "q_stack",
                    "specific_volume_stack",
                ),
            ),
            constitutive_law=model,
            et_benchmarks=et_benchmark,
        )
        t_stack = jnp.arange(0, solver.num_steps) * 0.0001

        solver = solver.setup()
        (
            p_stack,
            q_stack,
            specific_volume_stack,
        ) = hdx.run_et_solver(solver, 0.0001)

        plot_q_vs_p(
            ax.flat[0],
            p_stack=p_stack,
            q_stack=q_stack,
            xlim=(-1.0, 20.0),
            ylim=(-1.0, 30.0),
            color=model.other["color"],
            linestyle=model.other["ls"],
        )

plot_q_vs_p_M(
    ax.flat[0],
    models[0],
    (0, 100.0),
    color="red",
    xlim=(-1.0, 20.0),
)

fig.suptitle("TRX CU CD for MCC OCR=1 and OCR=3")
print("plotting..")
plt.show()

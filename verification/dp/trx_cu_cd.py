from cProfile import label
import os
from re import X
from turtle import color

from matplotlib import lines


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from all libraries (including JAX)
os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["JAX_DISABLE_JIT"] = "1"
# os.environ["EQX_ON_ERROR"] = "breakpoint"

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


p_0 = 150000.0

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
    hdx.TRX_CU(
        deps_zz_dt=4.0,
        p0=p_0,
        init_material_points=True,
        other=dict(type="TRX_CU"),
    ),
    hdx.TRX_CD(
        deps_zz_dt=4.0,
        p0=p_0,
        init_material_points=True,
        other=dict(type="TRX_CD"),
    ),
)

fig, ax = make_subplots()
for model in models:
    for et_benchmark in et_benchmarks:
        solver = hdx.ETSolver(
            material_points=hdx.MaterialPoints(
                p_stack=jnp.array([p_0]),
            ),
            num_steps=4000,
            config=hdx.Config(
                output=(
                    "p_stack",
                    "q_stack",
                    "specific_volume_stack",
                ),
            ),
            constitutive_law=model,
            et_benchmarks=et_benchmark,
        )

        solver = solver.setup()

        # print(solver.constitutive_law.get_critical_time(solver.material_points, 0.0025))
        (
            p_stack,
            q_stack,
            specific_volume_stack,
        ) = hdx.run_et_solver(solver, 0.0001)

        t_stack = jnp.arange(0, solver.num_steps) * 0.0001

        plot_q_vs_p(
            ax.flat[0],
            p_stack=p_stack,
            q_stack=q_stack,
            xlim=(-10000, 500000),
            ylim=(0, 500000),
            color=model.other["color"],
            linestyle=model.other["ls"],
        )


plot_q_vs_p_M(
    ax.flat[0],
    models[0],
    (0, 600000),
    color="red",
    xlim=(-10000, 500000),
    ylim=(0, 500000),
)

# fig.suptitle("TRX CU CD for MCC OCR=1 and OCR=3")
# print("plotting..")
plt.show()

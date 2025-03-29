from cProfile import label
import os
from re import X
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

fig.suptitle("TRX CU CD for MCC OCR=1 and OCR=3")
print("plotting..")
plt.show()


# plot_set1(
#     constitutive_law=models[1],
#     benchmark=et_benchmark,
#     fig_ax=(fig, ax),
#     plot_aux=True,
#     p_0=p_0,
# )

#         plot_set1(
#             p_stack=p_stack,
#             q_stack=q_stack,
#             t_stack=t_stack,
#             specific_volume_stack=specific_volume_stack,
#             constitutive_law=model,
#             benchmark=et_benchmark,
#             fig_ax=(fig, ax),
#         )
#         print(f"{model.other['label']} done..")


# # set limits
# ax.flat[0].set_xlim(0, 350)
# ax.flat[0].set_ylim(0, 350)


# ax.flat[1].set_xlim(75, 300)
# ax.flat[1].set_ylim(1.45, 1.5)
# ax.flat[1].set_xticks([100, 150, 350], minor=True)


# ax.flat[0].grid(True)
# ax.flat[1].grid(

#     True,
#     which="both",
# )


# def create_legend(fig, ncols=4):
#     lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
#     lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#     fig.legend(
#         lines,
#         labels,
#         ncols=ncols,
#         loc="outside lower center",
#     )
#     return fig


# create_legend(fig, 5)

# for i, label in enumerate(["(a)", "(b)"]):
#     ax.flat[i].set_title(label, y=0, pad=-35, verticalalignment="top")
# plt.savefig(f"{dir_path}/plots/et_solid_like_benchmarks.png")

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import hydraxmpm as hdx

dir_path = os.path.dirname(os.path.realpath(__file__))
import scienceplots
import matplotlib as mpl

plt.style.use("science")
mpl.rcParams["lines.linewidth"] = 1

p_0 = 500
constitutive_laws = (
    hdx.ModifiedCamClay(
        nu=0.2,
        M=1.2,
        R=1.0,
        lam=0.025,
        kap=0.005,
        ln_N=0.7,
        d=0.005,
        p_0=p_0,
        rho_p=1400,
        init_by_density=False,
    ),
    hdx.MuI_incompressible(
        mu_s=1.2 / jnp.sqrt(3),
        mu_d=1.9 / jnp.sqrt(3),
        I_0=0.279,
        K=50 * 2000 * 9.8 * 0.4,
        d=0.005,
        p_0=p_0,
        rho_p=1400,
        init_by_density=False,
        rho_0=812.0750143213877,
    ),
    # hdx.NewtonFluid(
    #     K=50 * 2000 * 9.8 * 0.4,
    #     # viscosity=0.001,
    #     viscosity=1e2,
    #     p_0=p_0,
    #     d=0.005,
    #     rho_p=1400,
    #     rho_0=812.07,
    #     init_by_density=False,
    # ),
)

et_benchmarks = (
    hdx.ConstantPressureSimpleShear(x=10.0, p=p_0, init_material_points=True),
    hdx.ConstantPressureSimpleShear(x=1.0, p=p_0, init_material_points=True),
    # hdx.ConstantVolumeSimpleShear(x=10.0, init_material_points=True),
    # hdx.ConstantVolumeSimpleShear(x=1.0, init_material_points=True),
)

title = "constant pressure shear element test"

# colors = ["red", "green", "blue"]
# cmap = plt.get_cmap()
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyles = ["-", "--", ":"]


def test_matrix():
    """Test to see if element test runs"""

    fig_multi, ax_multi = plt.subplots(
        ncols=3, nrows=2, figsize=(7.5 * 0.9, 4.5 * 0.9), dpi=300
    )
    # labels = ["MCC", "$\\mu (I)$-rheology"]
    labels = ["MCC", "$\\mu (I)$-rheology", "NF"]
    label_rates = [" ($\\dot\\gamma = 5.0$)", " ($\\dot\\gamma = 0.5$)"]

    for ci, model in enumerate(constitutive_laws):
        color = cycle[ci]
        for bi, benchmark in enumerate(et_benchmarks):
            label = labels[ci] + label_rates[bi]
            ls = linestyles[bi]
            solver = hdx.ETSolver(
                config=hdx.Config(
                    # num_steps=[100, 1000, 10000][bi],
                    # total_time=1.0,
                    total_time=[0.01, 0.1][bi],
                    dt=0.000001,
                    output=(
                        "p_stack",
                        "q_vm_stack",
                        "gamma_stack",
                        "dgammadt_stack",
                        "eps_v_stack",
                        "specific_volume_stack",
                        "inertial_number_stack",
                        "dgamma_p_dt_stack",
                        "deps_p_v_dt_stack",
                    ),
                ),
                constitutive_law=model,
                et_benchmarks=benchmark,
            )
            solver = solver.setup()

            out = solver.run()

            (
                p_stack,
                q_vm_stack,
                gamma_stack,
                dgammadt_stack,
                eps_v_stack,
                specific_volume_stack,
                inertial_number_stack,
                dgamma_p_dt_stack,
                deps_p_v_dt_stack,
            ) = out
            print(
                solver.constitutive_law.rho_0,
                solver.constitutive_law.rho_p,
                solver.material_points.rho_stack,
                dgammadt_stack.max(),
            )
            # print(
            #     dgammadt_stack.min(),
            #     dgammadt_stack.max(),
            #     eps_v_stack.min(),
            #     eps_v_stack.max(),
            #     # solver.config.num_steps,
            #     # solver.config.total_time,
            #     # solver.config.dt,
            # )
            # plots
            ax_multi.flat[0].plot(
                [0, p_0 * 1.1], [0, p_0 * 1.1 * 1.2], "-", c="red", lw=1, zorder=-1
            )
            ax_multi.flat[2].plot(
                [1.70, 1.73], [1.2, 1.2], "-", c="red", lw=1, zorder=-1
            )

            line_labels = (None, None)
            if (ci == 0) & (bi == 0):
                line_labels = ("ICL", "CSL")
            GAMMA = lambda ln_N: ln_N - (0.025 - 0.005) * jnp.log(2)  # noqa: E731
            ICL = lambda p: jnp.exp(0.7 - 0.025 * jnp.log(p))  # noqa: E731
            CSL = lambda p: jnp.exp(GAMMA(0.7) - 0.025 * jnp.log(p))  # noqa: E731
            ax_multi.flat[3].plot(
                [50, 2000],
                [ICL(50), ICL(2000)],
                "-",
                c="black",
                lw=1,
                zorder=-1,
                label=line_labels[0],
            )

            ax_multi.flat[3].plot(
                [50, 2000],
                [CSL(50), CSL(2000)],
                "-",
                c="red",
                lw=1,
                zorder=-1,
                label=line_labels[1],
            )
            ax_multi.flat[4].plot([0, 0.05], [1.2, 1.2], "-", c="red", lw=1, zorder=-1)
            ax_multi.flat[5].plot([0, 0.06], [1.2, 1.2], "-", c="red", lw=1, zorder=-1)

            fig_multi, ax_multi = hdx.plot_set1(
                p_stack,
                q_vm_stack,
                gamma_stack,
                dgammadt_stack,
                specific_volume_stack,
                inertial_number_stack,
                dgamma_p_dt_stack,
                deps_p_v_dt_stack,
                fig_ax=(fig_multi, ax_multi),
                color=color,
                ls=ls,
                label=label,
            )

            ax_multi.flat[3].set_ylim(1.68, 1.74)
            ax_multi.flat[3].set_xticks(
                [250, 500, 750],
                minor=True,
            )
            ax_multi.flat[3].set_xlim(250, 750)

            ax_multi.flat[3].tick_params(axis="x", which="minor", pad=5)
            # ax.tick_params(axis='y', which='major', pad=10)
            # ax.tick_params(axis='both', which='major', pad=10)
            ax_multi.flat[3].set_yticks([1.65, 1.7, 1.75], minor=True)
            ax_multi.flat[3].margins(0.17, 0.1)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig_multi.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig_multi.legend(lines, labels, ncols=3, bbox_to_anchor=[0.5, 0.0], loc="center")
    fig_multi.tight_layout()
    plt.subplots_adjust(bottom=0.17)

    fig_multi.savefig(dir_path + "/plots/et_matrix.png")


test_matrix()

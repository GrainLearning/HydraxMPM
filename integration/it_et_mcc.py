import os

import matplotlib.pyplot as plt
import jax
import hydraxmpm as hdx

dir_path = os.path.dirname(os.path.realpath(__file__))
import matplotlib as mpl
from matplotlib import ticker
import scienceplots

plt.style.use("science")
mpl.rcParams["lines.linewidth"] = 2


def test_single_et():
    """Test to see if element test runs"""
    jax.config.update("jax_enable_x64", True)
    solver = hdx.ETSolver(
        config=hdx.Config(
            total_time=1.0,
            dt=0.001,
            output=(
                "p_stack",
                "q_vm_stack",
                "gamma_stack",
                "dgammadt_stack",
                "specific_volume_stack",
                "inertial_number_stack",
            ),
        ),
        constitutive_law=hdx.ModifiedCamClay(
            nu=0.2,
            M=1.2,
            R=1.0,
            lam=0.026,
            kap=0.005,
            ln_N=0.7,
            d=0.005,
            p_0=1000,
            rho_p=1400,
            init_by_density=False,
        ),
        et_benchmarks=hdx.ConstantVolumeSimpleShear(
            x_range=(0.0, 0.1), y_range=(0.0, 0.0)
        ),
    )
    solver = solver.setup()

    out = solver.run()
    (
        p_stack,
        q_vm_stack,
        gamma_stack,
        dgammadt_stack,
        specific_volume_stack,
        inertial_number_stack,
    ) = out
    fig, ax = hdx.plot_set1(
        p_stack,
        q_vm_stack,
        gamma_stack,
        dgammadt_stack,
        specific_volume_stack,
        inertial_number_stack,
    )
    ax.flat[3].set_ylim(1.6, 1.8)
    ax.flat[3].set_xlim(450, 1150)
    ax.flat[3].margins(0.17, 0.15)
    ax.flat[3].set_xticks([500, 750, 1000], minor=True)
    ax.flat[3].set_yticks([1.6, 1.7, 1.8, 1.9], minor=True)

    fig.suptitle("Modified Cam-Clay simple shear")

    plt.tight_layout()

    plt.savefig(dir_path + "/plots/et_mcc_single_et.png")


def test_multiple_ets():
    """Test to see if element test runs"""
    jax.config.update("jax_enable_x64", True)
    et_benchmarks = (
        hdx.ConstantVolumeSimpleShear(x_range=(0.0, 0.1), y_range=(0.0, 0.0)),
        hdx.IsotropicCompression(x_range=(0.0, 0.002)),
        hdx.IsotropicCompression(x_range=(0.0, -0.001)),
    )
    # one plot together
    fig_multi, ax_multi = plt.subplots(ncols=3, nrows=2, figsize=(8, 5))
    for eti, et_benchmark in enumerate(et_benchmarks):
        # another indivudually
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(8, 5))
        solver = hdx.ETSolver(
            config=hdx.Config(
                total_time=1.0,
                dt=0.001,
                output=(
                    "p_stack",
                    "q_vm_stack",
                    "gamma_stack",
                    "dgammadt_stack",
                    "specific_volume_stack",
                    "inertial_number_stack",
                ),
            ),
            constitutive_law=hdx.ModifiedCamClay(
                nu=0.2,
                M=1.2,
                R=1.0,
                lam=0.026,
                kap=0.005,
                ln_N=0.7,
                d=0.005,
                p_0=1000,
                rho_p=1400,
                init_by_density=False,
            ),
            et_benchmarks=et_benchmark,
        )
        solver = solver.setup()

        out = solver.run()
        (
            p_stack,
            q_vm_stack,
            gamma_stack,
            dgammadt_stack,
            specific_volume_stack,
            inertial_number_stack,
        ) = out

        # plots
        fig_multi, ax_multi = hdx.plot_set1(
            p_stack,
            q_vm_stack,
            gamma_stack,
            dgammadt_stack,
            specific_volume_stack,
            inertial_number_stack,
            fig_ax=(fig_multi, ax_multi),
        )

        ax_multi.flat[3].set_ylim(1.65, 1.75)
        ax_multi.flat[3].set_xlim(450, 1150)
        ax_multi.flat[3].margins(0.17, 0.1)
        ax_multi.flat[3].set_xticks([500, 750, 1000], minor=True)
        ax_multi.flat[3].set_yticks([1.65, 1.7, 1.75], minor=True)

    fig_multi.suptitle(
        "Modified Cam-Clay simple shear + isotropic compression + isotropic extension"
    )

    fig_multi.tight_layout()

    fig_multi.savefig(dir_path + "/plots/et_mcc_multi_et_all.png")


test_single_et()
test_multiple_ets()

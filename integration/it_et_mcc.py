import os

import matplotlib.pyplot as plt
import jax
import hydraxmpm as hdx

dir_path = os.path.dirname(os.path.realpath(__file__))


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
            lam=0.0026,
            kap=0.0001,
            ln_N=0.7,
            d=0.005,
            p_0=1000,
            rho_p=1400,
            init_by_density=False,
        ),
        et_benchmarks=hdx.VolumeControlShear(x_range=(0.0, 0.5), y_range=(0.0, 0.0)),
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

    plt.tight_layout()

    plt.savefig(dir_path + "/plots/et_mcc_single_et.png")


def test_multiple_ets():
    """Test to see if element test runs"""
    jax.config.update("jax_enable_x64", True)
    et_benchmarks = (
        hdx.VolumeControlShear(x_range=(0.0, 0.2), y_range=(0.0, 0.0)),
        hdx.IsotropicCompression(x_range=(0.0, 0.0003)),
        hdx.IsotropicCompression(x_range=(0.0, -0.00003)),
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
                lam=0.0026,
                kap=0.0001,
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
            ax0_lim=[[-50, 1300], [-50, 750]],
            ax1_lim=[[-0.01, 0.12], [-50, 750]],
            ax2_lim=[[-0.01, 0.21], [-50, 750]],
            ax3_lim=[[400, 2000], [1.975, 1.985]],
            ax4_lim=[[-0.01, None], [None, None]],
            ax5_lim=[[-0.0001, None], [None, None]],
        )

        # fig, ax = hdx.plot_set1(
        #     p_stack,
        #     q_vm_stack,
        #     gamma_stack,
        #     dgammadt_stack,
        #     specific_volume_stack,
        #     inertial_number_stack,
        #     fig_ax=(fig, ax),
        #     ax0_lim=[[-100, 1300], [-100, 750]],
        #     ax1_lim=[[-0.01, 0.12], [-50, 750]],
        #     ax2_lim=[[-0.01, 0.21], [-50, 750]],
        #     ax3_lim=[[400, 2000], [1.95, 2.0]],
        #     ax4_lim=[[-0.01, None], [None, None]],
        #     ax5_lim=[[-0.0001, None], [None, None]],
        # )

        # fig.tight_layout()

        # fig.savefig(dir_path + f"/plots/et_mcc_multi_et_{eti}.png")

    fig_multi.tight_layout()

    fig_multi.savefig(dir_path + "/plots/et_mcc_multi_et_all.png")


# test_single_et()
test_multiple_ets()

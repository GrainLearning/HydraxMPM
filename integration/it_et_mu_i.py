import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
        constitutive_law=hdx.MuI_incompressible(
            mu_s=1.2 / jnp.sqrt(3),
            mu_d=1.5 / jnp.sqrt(3),
            I_0=0.279,
            K=50 * 2000 * 9.8 * 0.4,
            d=0.005,
            p_0=1000,
            rho_p=1400,
            init_by_density=False,
        ),
        et_benchmarks=hdx.ConstantVolumeSimpleShear(
            x_range=(0.0, 0.5), y_range=(0.0, 0.0)
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
        ax0_lim=[[-50, 1100], [-50, 1400]],
        ax1_lim=[[-0.01, 0.25], [-50, 1400]],
        ax2_lim=[[-0.01, 0.45], [-50, 1400]],
        # ax3_lim=[[1000, 1300], [1.38, 1.39]],
        ax4_lim=[[-0.01, None], [None, None]],
        ax5_lim=[[-0.0001, None], [None, None]],
    )

    plt.tight_layout()

    plt.savefig(dir_path + "/plots/et_mu_i_single_et.png")

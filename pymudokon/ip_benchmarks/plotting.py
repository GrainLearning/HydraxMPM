import matplotlib.pyplot as plt
import numpy as np

from .math_helpers import (
    get_gamma,
    get_pressure,
    get_q_vm,
    get_tau,
    get_volumetric_strain,
)


## modified Cam Clay yield surface
def get_a(Pt, beta, Pc):
    a = (Pc + Pt) / (1 + beta)
    return a


def get_b(Pt, beta, a, P):
    if Pt - a <= P:
        return 1.0
    return beta


def yield_surface_q_of_p(M, Pt, beta, Pc, P):
    a = get_a(Pt, beta, Pc)

    b = get_b(Pt, beta, a, P)

    # Yield surface as q of function p
    # (q/M)**2
    pow_q_M = a**2 - (1.0 / (b**2)) * (P - Pt + a) ** 2

    # taking q ~ P so positive roots only
    q = M * np.sqrt(pow_q_M)
    return q


def get_CSL_Line(P, M, Pt):
    q = -M * (P - Pt)
    return q


## end modified Cam Clay yield surface


def plot_strain_grid(
    strain_list,
    fig_ax=None,
    file=None,
    plot_options={},
    savefig_option={},
):
    if fig_ax is None:
        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    else:
        fig, axes = fig_ax

    num_steps = strain_list.shape[0]

    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                list(range(num_steps)),
                strain_list[:, i, j],
                **plot_options,
            )

            if i == 2:
                axes[i, j].set_xlabel("step")
            if j == 0:
                axes[i, j].set_ylabel(f"$\\varepsilon_{{{i+1}{j+1}}}$")

    plt.tight_layout()
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_stress_grid(
    stress_list,
    fig_ax=None,
    file=None,
    normalize_stress=1.0e6,
    plot_options={},
    savefig_option={},
):
    if fig_ax is None:
        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    else:
        fig, axes = fig_ax

    units = "(Pa)"
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    num_steps = stress_list.shape[0]

    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                list(range(num_steps)),
                stress_list[:, i, j] / normalize_stress,
                **plot_options,
            )

            if i == 2:
                axes[i, j].set_xlabel("step")
            if j == 0:
                axes[i, j].set_ylabel(rf"$\sigma_{{{i+1}{j+1}}}$ {units}")

    plt.tight_layout()
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_q_p(
    stress_list,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    scatter=False,
    plot_options={},
    savefig_option={},
):
    pressure_list = get_pressure(stress_list)
    q_vm_list = get_q_vm(stress_list)

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    if scatter:
        ax.scatter(
            pressure_list / normalize_stress,
            q_vm_list / normalize_stress,
            **plot_options,
        )
    else:
        ax.plot(
            pressure_list / normalize_stress,
            q_vm_list / normalize_stress,
            **plot_options,
        )

    units = "(Pa)"
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel(f"$p$ {units}")
    ax.set_ylabel(f"$q$ {units}")

    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_q_eps1(
    stress_list,
    strain_list,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    q_vm_list = get_q_vm(stress_list)

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    ax.plot(
        -strain_list[:, 0, 0],
        q_vm_list / normalize_stress,
        **plot_options,
    )

    units = "(Pa)"
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel("$\\varepsilon_1$ (-)")
    ax.set_ylabel(f"$q$ {units}")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_eps_v_eps1(
    strain_list,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    eps_v_list = get_volumetric_strain(strain_list)

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    ax.plot(
        -strain_list[:, 0, 0],
        eps_v_list,
        **plot_options,
    )

    ax.set_xlabel("$\\varepsilon_1$ (-)")
    ax.set_ylabel("$\\varepsilon_v$ (-)")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_v_lnp(
    stress_list,
    strain_list,
    v0=2.0,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    scatter=False,
    plot_options={},
    savefig_option={},
):
    eps_v_list = get_volumetric_strain(strain_list)
    pressure_list = get_pressure(stress_list)

    v_list = []
    v_curr = v0
    for eps_v in eps_v_list:
        v_curr = v0 * (1.0 - eps_v)
        v_list.append(v_curr)
        # if eps_v < 0.0:
        # print("Negative volumetric strain")
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    if scatter:
        ax.scatter(
            np.log(pressure_list / normalize_stress),
            v_list,
            **plot_options,
        )
    else:
        ax.plot(
            np.log(pressure_list / normalize_stress),
            v_list,
            **plot_options,
        )

    units = "(Pa)"
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel(f"ln $p$ {units}")
    ax.set_ylabel("Specific volume $v$ (-)")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_tau_gamma(
    stress_list,
    strain_list,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    tau_list = get_tau(stress_list)

    gamma_list = get_gamma(strain_list)

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    ax.plot(
        gamma_list,
        tau_list / normalize_stress,
        **plot_options,
    )

    units = "(Pa)"
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel(r"$\gamma$ (-)")
    ax.set_ylabel(f"$\\tau$ {units}")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_solid_volume_fraction_pressure(
    stress_list,
    solid_volume_fraction_list,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    scatter=False,
    plot_options={},
    savefig_option={},
):
    pressure_list = get_pressure(stress_list)

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    if scatter:
        ax.scatter(
            pressure_list / normalize_stress,
            solid_volume_fraction_list,
            **plot_options,
        )
    else:
        ax.plot(
            pressure_list / normalize_stress,
            solid_volume_fraction_list,
            **plot_options,
        )

    units = "(Pa)"
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel("$p$")
    ax.set_ylabel(rf"$\phi$ {units}")

    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_sigma1_eps1(
    stress_list,
    strain_list,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    import matplotlib.pyplot as plt

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    ax.plot(
        -strain_list[:, 0, 0],
        -stress_list[:, 0, 0] / normalize_stress,
        **plot_options,
    )
    units = "(Pa)"
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_ylabel(rf"$\sigma_1$ {units}")
    ax.set_xlabel("$\\varepsilon_1$ (-)")
    if file is not None:
        plt.savefig(file, **savefig_option)

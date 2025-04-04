import os
from re import L


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from all libraries (including JAX)
os.environ["JAX_PLATFORMS"] = "cpu"


import hydraxmpm as hdx
import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt


import numpy as np
import jax.numpy as jnp

from functools import partial
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

import matplotlib.pyplot as plt
import matplotlib as mpl

import scienceplots

plt.style.use(["science", "no-latex"])


import equinox as eqx
import os


dir_path = os.path.dirname(os.path.realpath(__file__))


mpl.rcParams["lines.linewidth"] = 2

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def safe_relative_l2_loss(y_pred, y_true, epsilon=1e-12):
    return jnp.mean((y_pred - y_true) ** 2 / (y_true**2 + epsilon))


def run_mcc(M, lam, solver: hdx.SIPSolver):
    new_solver1 = eqx.tree_at(
        lambda state: (state.constitutive_law.M, state.constitutive_law.lam),
        solver,
        (M, lam),
    )

    ln_v_stack = new_solver1.constitutive_law.get_ln_v0(
        new_solver1.material_points.p_stack
    )

    # current density and slope of icl is coupled
    rho_stack = new_solver1.constitutive_law.rho_p / jnp.exp(ln_v_stack)

    new_material_points = new_solver1.material_points.init_mass_from_rho_0(rho_stack)

    new_solver2 = eqx.tree_at(
        lambda state: (state.material_points),
        new_solver1,
        (new_material_points),
    )

    return new_solver2.run(dt=dt)


@jax.jit
def run_mcc_loss(M, lam, solver: hdx.SIPSolver, target_q_p_stack: jnp.array):
    (sim_q_p_stack,) = run_mcc(M, lam, solver)

    loss = jax.vmap(safe_relative_l2_loss)(sim_q_p_stack, target_q_p_stack).sum()

    return loss


p_0 = 1000
fric_angle = 19.8
fric_angle_rad = jnp.deg2rad(fric_angle)
rho_0 = 2650.0


# matching Mohr-Coulomb criterion under triaxial extension conditions
mu = (
    6 * jnp.sin(fric_angle_rad) / (jnp.sqrt(3) * (3 + jnp.sin(jnp.deg2rad(fric_angle))))
)

num_samples_sqrt = 6
quiver_density = 1
example_index = (3, 4)

M = mu * jnp.sqrt(3)
lam = 0.0058
run_models = [run_mcc]
run_loss = [run_mcc_loss]
models = (
    hdx.ModifiedCamClay(
        nu=0.3,
        M=M,
        lam=lam,
        kap=0.0012,
        R=1.0,
        rho_0=rho_0,
        rho_p=rho_0,
        settings=dict(atol=1e-3),
        other=dict(
            parameter_names=("M", "lam"),
            parameter_bounds=((M * 0.85, M * 1.15), (lam / 2, lam * 4)),
            parameter_targets=(M, lam),
        ),
    ),
)

dgamma_dt_start = 0.5


x_slow = dgamma_dt_start * 2


num_steps = 2500
x1_stack = jnp.ones(num_steps) * x_slow


sip_benchmark = hdx.S_CD(
    deps_xy_dt=x1_stack,
    num_steps=num_steps,
    p0=p_0,
    init_material_points=True,
    other=dict(type="S_CD"),
)


dt = 0.0001
# scale for the quiver plot for visualization
qscale = 1e4

t_stack = jnp.arange(0, sip_benchmark.num_steps) * dt


for mi, model in enumerate(models):
    run_model = run_models[mi]
    run_loss = run_loss[mi]

    solver = hdx.SIPSolver(
        material_points=hdx.MaterialPoints(
            p_stack=jnp.array([p_0]),
        ),
        output_dict=("q_p_stack",),
        constitutive_law=model,
        sip_benchmarks=sip_benchmark,
    )

    solver = solver.setup()
    jitted_run_model = jax.jit(run_model)

    (target_q_p_stack,) = jitted_run_model(
        model.other["parameter_targets"][0], model.other["parameter_targets"][1], solver
    )

    param_1 = jnp.linspace(
        model.other["parameter_bounds"][0][0],
        model.other["parameter_bounds"][0][1],
        num_samples_sqrt,
    )

    param_2 = jnp.linspace(
        model.other["parameter_bounds"][1][0],
        model.other["parameter_bounds"][1][1],
        num_samples_sqrt,
    )

    p1_mesh, p2_mesh = jnp.meshgrid(param_1, param_2)

    eps_p1 = param_1.at[1].get() - param_1.at[0].get()
    eps_p2 = param_2.at[1].get() - param_2.at[0].get()

    def finite_forward_diff(p1, p2):
        f_p1_perturbed = run_loss(p1 + eps_p1, p2, solver, target_q_p_stack)
        grad_p1 = (f_p1_perturbed - run_loss(p1, p2, solver, target_q_p_stack)) / eps_p1

        f_p2_perturbed = run_loss(p1, p2 + eps_p2, solver, target_q_p_stack)
        grad_p2 = (f_p2_perturbed - run_loss(p1, p2, solver, target_q_p_stack)) / eps_p2

        return grad_p1, grad_p2

    jit_get_loss = jax.jit(run_loss)
    jit_finite_diff = jax.jit(finite_forward_diff)

    ffdLdp1_stack, ffdLdp2_stack = jax.vmap(jit_finite_diff)(
        p1_mesh.reshape(-1), p2_mesh.reshape(-1)
    )

    L_stack = jax.vmap(jit_get_loss, in_axes=(0, 0, None, None))(
        p1_mesh.reshape(-1), p2_mesh.reshape(-1), solver, target_q_p_stack
    )

    dLdp1_stack, dLdp2_stack = jax.vmap(
        jax.jacfwd(jit_get_loss, argnums=(0, 1)), in_axes=(0, 0, None, None)
    )(p1_mesh.reshape(-1), p2_mesh.reshape(-1), solver, target_q_p_stack)

    p1_example = p1_mesh[example_index]
    p2_example = p2_mesh[example_index]

    (example_q_p_stack,) = jitted_run_model(p1_example, p2_example, solver)

    print("done plotting...")
    fig, ax = plt.subplots(
        ncols=3,
        figsize=(8, 3),
        dpi=300,
        layout="constrained",
    )

    ax.flat[0].plot(
        t_stack,
        target_q_p_stack,
        color=cycle[0],
        label="target",
    )

    ax.flat[0].plot(
        t_stack,
        example_q_p_stack,
        label="sample ($M$"
        + f"={p1_example:.2f},   "
        + "$\\lambda$"
        + f"={p2_example:.2f} )",
        color=cycle[5],
    )

    ax.flat[0].grid(True, linestyle="--", alpha=0.5)  # add grid.
    ax.flat[0].set_ylabel("$q/p$ [-]")
    ax.flat[0].set_xlabel("$t$ [s]")

    contour = ax.flat[1].contourf(
        p1_mesh,
        p2_mesh,
        L_stack.reshape(p1_mesh.shape),
        cmap="plasma",
        levels=20,
    )

    ax.flat[1].set_ylabel("$\\lambda$ [-]")
    ax.flat[1].set_xlabel("M [-]")

    U_ad = -dLdp1_stack * model.other["parameter_targets"][0]
    V_ad = -dLdp2_stack * model.other["parameter_targets"][1]

    ax.flat[1].plot([p1_example], [p2_example], "^", color=cycle[5], markersize=8)
    ax.flat[1].plot(
        [model.other["parameter_targets"][0]],
        [model.other["parameter_targets"][1]],
        "X",
        color=cycle[0],
        markersize=8,
    )

    ax.flat[1].quiver(
        p1_mesh.reshape(-1)[::quiver_density],
        p2_mesh.reshape(-1)[::quiver_density],
        U_ad[::quiver_density],
        V_ad[::quiver_density],
        scale=qscale,
        width=0.008,
        angles="xy",
        scale_units="xy",
        headwidth=3.5,
        color="black",
        pivot="tip",
    )
    ax.flat[1].grid(True, linestyle="--", alpha=0.5)

    ### finite difference
    contour = ax.flat[2].contourf(
        p1_mesh,
        p2_mesh,
        L_stack.reshape(p1_mesh.shape),
        cmap="plasma",
        levels=20,
    )

    cbar = fig.colorbar(
        contour, ax=ax.flat[2], label="$L$ [-]", orientation="vertical"
    )  # vertical colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.label.set_rotation(270)
    cbar.ax.yaxis.labelpad = 15

    ax.flat[2].set_ylabel("$\\lambda$ [-]")
    ax.flat[2].set_xlabel("M [-]")

    norm_fd = jnp.sqrt(ffdLdp1_stack**2 + ffdLdp2_stack**2 + 1e-9)
    U_fd = -ffdLdp1_stack
    V_fd = -ffdLdp2_stack

    U_fd = U_fd * model.other["parameter_targets"][0]
    V_fd = V_fd * model.other["parameter_targets"][1]
    ax.flat[2].quiver(
        p1_mesh.reshape(-1)[::quiver_density],
        p2_mesh.reshape(-1)[::quiver_density],
        U_fd[::quiver_density],
        V_fd[::quiver_density],
        scale=qscale,
        width=0.008,
        angles="xy",
        scale_units="xy",
        headwidth=3.5,
        color="black",
        pivot="mid",
    )

    ax.flat[2].plot([p1_example], [p2_example], "^", color=cycle[5], markersize=8)
    ax.flat[2].plot(
        [model.other["parameter_targets"][0]],
        [model.other["parameter_targets"][1]],
        "X",
        color=cycle[0],
        markersize=8,
    )
    ax.flat[2].grid(True, linestyle="--", alpha=0.5)

    def create_legend(fig, ncols=4):
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(
            lines,
            labels,
            ncols=ncols,
            loc="outside lower center",
        )
        return fig

    create_legend(fig, 5)

    for i, label in enumerate(["(a)", "(b)", "(c)"]):
        ax.flat[i].set_title(label, y=0, pad=-35, verticalalignment="top")

    plt.savefig(dir_path + "/plots/sip_mcc_gradients.png")

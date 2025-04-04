import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from all libraries (including JAX)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

import hydraxmpm as hdx

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import matplotlib as mpl

import scienceplots

plt.style.use(["science", "no-latex"])


import equinox as eqx
import os

jax.config.update("jax_enable_x64", True)


dir_path = os.path.dirname(os.path.realpath(__file__))


mpl.rcParams["lines.linewidth"] = 2

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


p_0 = 1000.0

fric_angle = 19.8
fric_angle_rad = jnp.deg2rad(fric_angle)
rho_0 = 2650.0
rho_p = 2650.0

K = 7e5  # [Pa]

# matching Mohr-Coulomb criterion under triaxial extension conditions
mu = (
    6 * jnp.sin(fric_angle_rad) / (jnp.sqrt(3) * (3 + jnp.sin(jnp.deg2rad(fric_angle))))
)


models = (
    hdx.ModifiedCamClay(
        nu=0.3,
        M=mu * jnp.sqrt(3),
        lam=0.0058,
        kap=0.0012,
        R=1.0,
        rho_0=rho_0,
        rho_p=rho_p,
        settings=dict(atol=1e-3),
        other=dict(label="Modified Cam-Clay", zorder=-1, color=cycle[0]),
    ),
    hdx.DruckerPrager(
        nu=0.3,
        K=K,
        mu_1=mu,
        rho_0=rho_0,
        rho_p=rho_p,
        other=dict(label="Drucker-Prager", zorder=-1, color=cycle[2]),
    ),
    hdx.MuI_incompressible(
        other=dict(label="$ \\mu (I)$-rheology", ls="-", zorder=-1, color=cycle[1]),
        mu_s=mu,
        mu_d=1.9,
        I_0=0.279,
        K=K,
        d=0.00125,
        rho_p=rho_p,
        rho_0=rho_0,
    ),
    hdx.NewtonFluid(
        other=dict(label="Newton Fluid", ls="-", zorder=0, color=cycle[4]),
        K=K,
        viscosity=0.002,
        alpha=7.0,
        rho_0=rho_0,
    ),
)


dgamma_dt_start = 0.5
dgamma_dt_end = 50.0

print(dgamma_dt_start, dgamma_dt_end)

x_slow = dgamma_dt_start * 2
x_fast = dgamma_dt_end * 2

num_steps_s1 = 25000
x1_stack = jnp.ones(num_steps_s1) * x_slow


num_steps_s2 = 5000
x2_stack = jnp.linspace(x_slow, x_fast, num_steps_s2)


num_steps_s3 = 15000
x3_stack = jnp.ones(num_steps_s3) * x_fast


x_total_stack = jnp.concat((x1_stack, x2_stack, x3_stack))

sip_benchmarks = (
    hdx.S_CD(
        deps_xy_dt=jnp.concat((x1_stack, x2_stack, x3_stack)),
        # deps_xy_dt=4.0,
        num_steps=x_total_stack.shape[0],
        p0=p_0,
        init_material_points=True,
        other=dict(type="S_CD"),
    ),
)


fig, axes = plt.subplots(
    ncols=3,
    figsize=(8, 3),
    dpi=300,
    layout="constrained",
)
dt = 0.00001

for mi, model in enumerate(models):
    for ie, sip_benchmark in enumerate(sip_benchmarks):
        solver = hdx.SIPSolver(
            material_points=hdx.MaterialPoints(
                p_stack=jnp.array([p_0]),
            ),
            output_dict=("p_stack", "q_stack", "eps_v_stack", "specific_volume_stack"),
            constitutive_law=model,
            sip_benchmarks=sip_benchmark,
        )
        t_stack = jnp.arange(0, sip_benchmark.num_steps) * dt

        solver = solver.setup()

        (p_stack, q_stack, eps_v_stack, specific_volume_stack) = solver.run(dt=dt)

        if ie == 0:
            label = model.other.get("label", "")
        else:
            label = None

        print(eps_v_stack)
        p_stack = p_stack
        q_stack = q_stack

        hdx.make_plot(
            axes.flat[0],
            p_stack,
            q_stack,
            xlabel="$p$ [Pa]",
            ylabel="$q$ [Pa]",
            color=model.other.get("color", "black"),
            label=label,
            ls=sip_benchmark.other.get("ls", "-"),
            zorder=model.other.get("zorder", 1),
            start_end_markersize=7,
        )

        hdx.make_plot(
            axes.flat[1],
            t_stack,
            q_stack / p_stack,
            xlabel="$t$ [s]",
            ylabel="$q/p$ [-]",
            color=model.other["color"],
            ls="-",
            zorder=model.other["zorder"],
            start_end_markersize=7,
        )

        hdx.make_plot(
            axes.flat[2],
            t_stack,
            eps_v_stack,
            xlabel="$t$ [s]",
            ylabel="$\\varepsilon_v$ [-]",
            color=model.other["color"],
            ls="-",
            zorder=model.other["zorder"],
            start_end_markersize=7,
        )
        print(f"{model.other['label']} {sip_benchmark.other['type']} done..")


# plot CSL
p_aux = jnp.arange(0, 1500, 10)
q_csl = models[0].CSL_q_p(p_aux)

axes.flat[0].plot(p_aux, q_csl, "r-", lw=1.0, zorder=-1, label="CSL")

t_aux = jnp.arange(0, 0.5, 0.01)
axes.flat[1].plot(t_aux, jnp.ones_like(t_aux) * models[0].M, "r-", lw=1.0, zorder=-3)


# set limits
axes.flat[0].margins(0.05)
axes.flat[1].margins(0.05)
axes.flat[2].margins(0.05)


axes.flat[0].grid(True)
axes.flat[1].grid(True)
axes.flat[2].grid(True)


def create_legend(fig, ncols=5):
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, ncols=ncols, loc="outside lower center")
    return fig


create_legend(fig, 5)

for i, label in enumerate(["(a)", "(b)", "(c)"]):
    axes.flat[i].set_title(label, y=0, pad=-35, verticalalignment="top")
fig.savefig(dir_path + "/plots/sip_pressure_control.png")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from all libraries (including JAX)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

import matplotlib.pyplot as plt

from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl


from matplotlib.animation import FuncAnimation, PillowWriter

import scienceplots


plt.style.use(["science", "no-latex"])


import os

jax.config.update("jax_enable_x64", True)


dir_path = os.path.dirname(os.path.realpath(__file__))


mpl.rcParams["lines.linewidth"] = 2.5
mpl.rcParams["font.size"] = 16

mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["axes.facecolor"] = "none"
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["savefig.dpi"] = 150
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
        name="Modified Cam-Clay",
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
        name="Drucker-Prager",
        nu=0.3,
        K=K,
        mu_1=mu,
        rho_0=rho_0,
        rho_p=rho_p,
        other=dict(label="Drucker-Prager", zorder=-1, color=cycle[2]),
    ),
    hdx.MuI_incompressible(
        name="Mu(I) Incompressible",
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
        name="Newtonian Fluid",
        other=dict(label="Newtonian Fluid", ls="-", zorder=0, color=cycle[4]),
        K=K,
        viscosity=0.002,
        alpha=7.0,
        rho_0=rho_0,
    ),
)
dt = 1e-5
dgamma_dt_start = 0.05
dgamma_dt_end = 50.0

x_slow = dgamma_dt_start
x_fast = dgamma_dt_end

num_steps_s1 = 200000
x1_stack = jnp.ones(num_steps_s1) * x_slow


num_steps_s2 = 200000
x2_stack = jnp.linspace(x_slow, x_fast, num_steps_s2)


num_steps_s3 = 200000
x3_stack = jnp.ones(num_steps_s3) * x_fast


x_total_stack = jnp.concat((x1_stack, x2_stack, x3_stack))

num_steps = x_total_stack.shape[0]

sip_benchmarks = (
    hdx.ConstantPressureShear(
        deps_xy_dt=jnp.concat((x1_stack, x2_stack, x3_stack)),
        num_steps=num_steps,
        p0=p_0,
        init_material_points=True,
        other=dict(type="ConstantPressureShear"),
    ),
)

time_stack = jnp.arange(0, x_total_stack.shape[0]) * dt


output_vars = ("p_stack", "q_stack", "eps_v_stack", "specific_volume_stack")

# --- Model Specific Metadata ---
output_data = {}
artists = {}
for i, model in enumerate(models):
    output_data[model.name] = {}
    artists[model.name] = {}

print("Starting simulation ...")
# --- Generate The data ---
for mi, model in enumerate(models):
    for ie, sip_benchmark in enumerate(sip_benchmarks):
        solver = hdx.SIPSolver(
            material_points=hdx.MaterialPoints(p_stack=jnp.array([p_0])),
            output_vars=output_vars,
            constitutive_law=model,
            sip_benchmarks=sip_benchmark,
        )

        solver = solver.setup()

        sim_output = solver.run(dt=dt)

        for i, key in enumerate(output_vars):
            output_data[model.name][key] = np.array(sim_output[i])

    print("{} {} done..".format(model.other["label"], sip_benchmark.other["type"]))
print("Simulation done, creating plots ...")


# --- Initialize the plots ---
fig, axes = plt.subplots(
    ncols=3,
    figsize=(12, 3),
    layout="constrained",
)


def plot_line(ax, x, y, model):
    return ax.plot(
        x,
        y,
        color=model.other["color"],
        ls="-",
        label=model.other.get("label", ""),
        zorder=model.other.get("zorder", 1),
    )[0]


def plot_marker(ax, x, y, model, marker):
    return ax.plot(x, y, marker, color=model.other["color"], markersize=7, zorder=10)[0]


for model in models:
    artists[model.name] = {
        "line_qp": plot_line(axes[0], [], [], model),
        "marker_qp_start": plot_marker(axes[0], [], [], model, "o"),
        "line_eps_v": plot_line(axes[1], [], [], model),
        "marker_eps_v_start": plot_marker(axes[1], [], [], model, "o"),
        "line_pq": plot_line(axes[2], [], [], model),
        "marker_pq_start": plot_marker(axes[2], [], [], model, "o"),
    }


axes[0].set(
    xlabel="t [s]",
    ylabel="$q/p$ [-]",
    xticks=[0, 2, 4, 6],
    yticks=[0.0, 0.3, 0.6, 0.9, 1.2],
    xlim=(-0.2, time_stack.max() + 0.2),
    ylim=(-0.2, 1.2),
)
axes[1].set(
    xlabel="$t$ [s]",
    ylabel="$\\varepsilon_v$ [%]",
    xlim=(-0.2, time_stack.max() + 0.2),
    ylim=(-0.05, 0.40),
    xticks=[0, 2, 4, 6],
    yticks=[0.0, 0.1, 0.2, 0.3, 0.4],
)
axes[2].set(
    xlabel="$p$ [Pa]",
    ylabel="$q$ [Pa]",
    xlim=(0, 2000),
    ylim=(-50, 2000),
    xticks=[0, 500, 1000, 1500, 2000],
    yticks=[0, 500, 1000, 1500, 2000],
)
# --- Add Legend ---
# Collect handles and labels from one of the axes (e.g., axes[0])
# Or create a figure-level legend
handles, labels = axes[0].get_legend_handles_labels()
# Place legend outside the plot area
legend = fig.legend(
    handles,
    labels,
    loc="outside lower center",
    frameon=False,
    ncols=6,
)  # frameon=False for cleaner transparency


# --- animate plots ---


def init_anim():
    """Initialize animation with empty data."""
    for model in models:
        model_name = model.name
        for art in artists[model_name].values():
            for a in art:
                a.set_data([], [])
    return [
        a for model_art in artists.values() for art in model_art.values() for a in art
    ]


# Downsample frames for animation
downsample_step = 10000  # Adjust based on your needs
total_frames = num_steps // downsample_step


def animate(i):
    """Update frame for animation."""
    i = i * downsample_step + 1  # Apply downsampling
    updated_artists = []

    for model in models:
        data = output_data[model.name]

        p = data["p_stack"]
        q = data["q_stack"]
        eps_v = data["eps_v_stack"] * 100

        # Update q/p plot
        artists[model.name]["line_qp"].set_data(time_stack[:i], (q / p)[:i])
        artists[model.name]["marker_qp_start"].set_data(time_stack[:1], (q / p)[:1])
        updated_artists.extend(
            [
                artists[model.name]["line_qp"],
                artists[model.name]["marker_qp_start"],
            ]
        )

        # Update eps_v plot
        artists[model.name]["line_eps_v"].set_data(time_stack[:i], eps_v[:i])
        artists[model.name]["marker_eps_v_start"].set_data(time_stack[:1], eps_v[:1])
        updated_artists.extend(
            [
                artists[model.name]["line_eps_v"],
                artists[model.name]["marker_eps_v_start"],
            ]
        )

        # Update p-q plot
        artists[model.name]["line_pq"].set_data(p[:i], q[:i])
        artists[model.name]["marker_pq_start"].set_data(p[:1], q[:1])
        updated_artists.extend(
            [
                artists[model.name]["line_pq"],
                artists[model.name]["marker_pq_start"],
            ]
        )

    return updated_artists


# --- Helper function to set colors ---
def set_plot_colors(fig, axes, legend, color):
    """Sets color for text elements: labels, ticks, legend."""
    print(f"Setting plot element colors to: {color}")
    fig.suptitle("Constant Pressure Shear", fontsize=14, color=color)
    # Set legend text color
    if legend:
        for text in legend.get_texts():
            text.set_color(color)

    for ax in axes:
        ax.patch.set_alpha(0.0)
        # Axis labels
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)

        # Tick parameters (sets color for ticks and labels)
        ax.tick_params(axis="x", colors=color)
        ax.tick_params(axis="y", colors=color)

        # Spines (the box lines)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)

        # Optional: Axis title if exists
        # ax.title.set_color(color)


# Create animation
ani = FuncAnimation(
    fig,
    animate,
    frames=total_frames,
    interval=50,
    repeat=True,
)

# Save or show animation
set_plot_colors(fig, axes, legend, "black")
fig.canvas.draw_idle()
ani.save(
    dir_path + "/plots/sip_animation_dark.gif",
    writer=PillowWriter(fps=25),
    dpi=150,
    savefig_kwargs={"transparent": True, "facecolor": fig.get_facecolor()},
)

set_plot_colors(fig, axes, legend, "white")
fig.canvas.draw_idle()
ani.save(
    dir_path + "/plots/sip_animation_light.gif",
    writer=PillowWriter(fps=25),
    dpi=150,
    savefig_kwargs={"transparent": True, "facecolor": fig.get_facecolor()},
)

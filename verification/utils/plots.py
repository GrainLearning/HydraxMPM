import matplotlib.pyplot as plt
from matplotlib.ticker import (
    LogLocator,
    ScalarFormatter,
    FormatStrFormatter,
    NullFormatter,
    StrMethodFormatter,
)

from matplotlib import ticker
import jax.numpy as jnp

import matplotlib.patheffects as mpe
import hydraxmpm as hdx

import matplotlib as mpl

mpl.rcParams["lines.linewidth"] = 1.5


def make_subplots(ncols=4, nrows=2, figsize=(12, 6)):
    fig, ax = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        dpi=300,
        layout="constrained",
    )
    return fig, ax


def plot_q_vs_p(ax, q_stack, p_stack, **kwargs):
    hdx.make_plot(ax, p_stack, q_stack, **kwargs)
    ax.set_xlabel(r"$p$ (Pa)")
    ax.set_ylabel(r"$q$ (Pa)")

    ax.grid(True)


def plot_q_vs_p_M(ax, model, p_bounds, **kwargs):
    hdx.make_plot(
        ax, p_bounds, (0, p_bounds[1] * model.M), start_end_markers=True, **kwargs
    )
    ax.set_xlabel(r"$p$ (Pa)")
    ax.set_ylabel(r"$q$ (Pa)")
    ax.grid(True)

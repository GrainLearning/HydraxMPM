import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

import plotting as ps

from UEPI_MCC import get_v_ncl


###########################################################################
# Load data
###########################################################################


def load_data(file_name: str):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df = pd.read_excel(
        file_path, 
        usecols="F,L,N,S,T", 
        skiprows=11, 
        names=["eps_vol", "time", "p", "stress_rate","void_ratio"],
        sheet_name="Data"
    )

    # Filter and process
    df = df[20:].reset_index(drop=True)
    df["specific_volume"] = df["void_ratio"] + 1.0
    
    return df

###########################################################################
# Plotting
###########################################################################


def plot_results(df, lam, N, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    
    fig = plt.figure(figsize=ps.FIG_SIZES["single_col"])

    ax = fig.add_subplot(1, 1, 1)

    p_fit = np.logspace(0, 3, 100)
    
    # note pressures are in kPa, so p_ref= 1.0 is 1000 Pa
    v_pred = get_v_ncl(p_fit, 1.0, N, lam)

    ax.plot(p_fit, v_pred, lw=2.0, color="tab:orange",
        **{
            "label": f"NCL"
        }
        )

    ax.plot(
        df["p"], df["specific_volume"],
        marker='o', markeredgecolor="black", color="tab:green",
        markevery=10, alpha=0.6, markersize=4, ls="",
        label="S{\o}rensen et al., 2007", zorder=-1
    )

    ps.annotate_axes(ax, "(a)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlim(10, 525)
    ax.set_xticks([10, 20, 50, 100, 200, 500])
    ax.set_ylim(1.8, 2.4)
    ax.set_yticks([1.8,2.0,2.2,2.4])

    ax.set_yticks([], minor=True)


    formatter = mpl.ticker.ScalarFormatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(ps.LABELS["pressure"])
    ax.set_ylabel(ps.LABELS["specific_volume"])

    handles, labels = ps.sort_legend_handles([ax])

    sorted_pairs = sorted(zip(handles, labels), key=lambda x: 0 if "NCL" in x[1] else 1)
    handles, labels = zip(*sorted_pairs)

    ax.legend(handles, labels, loc='lower left')


    out_path = os.path.join(plot_dir, "iso_london.pdf")
    fig.savefig(out_path, bbox_inches="tight",dpi=300)
    print(f"Plot saved to: {out_path}")

###########################################################################
# Main script
###########################################################################


if __name__ == "__main__":
    
    df = load_data("data/S2LCrA2 - iso comp.xls")

    df_low_rate = df[df["stress_rate"] < 2.].reset_index(drop=True)

    p_data = df_low_rate["p"].values
    v_data = df_low_rate["specific_volume"].values

    # note pressures are in kPa, so p_ref= 1.0 is 1000 Pa
    fit_ncl = lambda p,  N, lam: get_v_ncl(p, 1.0, N, lam)
    
    popt, pcov = curve_fit(
        fit_ncl, 
        p_data, 
        v_data, 
        p0=[3.0,0.1],
        bounds=([0.0, 0.0], [np.inf, np.inf]) 
    )

    N_opt, lam_opt = popt
    

    print(f"Scipy Optimization Results:")
    print(f"Lambda : {lam_opt:.5f}")
    print(f"N      : {N_opt:.5f}")

    plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "figures")
    plot_results(df, lam_opt, N_opt, plot_dir)
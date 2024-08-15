
import jax.numpy as jnp
import chex
from typing import Tuple,Dict
from functools import partial
from .plot import make_plots, PlotHelper
from ..utils.math_helpers import (
    get_pressure_stack,
    get_q_vm_stack,phi_to_e_stack,
    get_sym_tensor_stack,
    get_volumetric_strain_stack,
    get_scalar_shear_strain_stack,
    get_hencky_strain_stack
)


def plot_set1(
    stress_stack: chex.Array,
    phi_stack: chex.Array,
    L_stack: chex.Array,
    plot_helper_args: Dict = None,
    fig_ax: Tuple = None
    )-> Tuple:
    """Create the plot set 1.
    
    Plots the following:
    
    q vs p | e-log p | M - e 
    deps_v_dt vs dgamma_dt | log p - phi |  M - phi
     
    Args:
        stress_stack (chex.Array): list of stress tensors
        phi_stack (chex.Array): list of solid volume fractions
        L_stack (chex.Array): list of velocity gradients
        fig_ax (Tuple, optional): Tup. Defaults to None.

    Returns:
        Tuple: Fig axis pair
    """
    
    # pass arguments to plot helper from outside
    if plot_helper_args is None:
        plot_helper_args={}
    
    _PlotHelper = partial(PlotHelper,**plot_helper_args)
    
    # Plot 1: q vs p
    p_stack = get_pressure_stack(stress_stack)

    q_stack = get_q_vm_stack(stress_stack)

    plot1_qp = _PlotHelper(
        x=p_stack,
        y=q_stack,
        xlabel="$p$ [Pa]",
        ylabel="$q$ [Pa]",
        xlim=[0, p_stack.max()*1.2],
        ylim=[0, q_stack.max()*1.2],
    )
    
    # Plot 2: e-log p
    e_stack = phi_to_e_stack(phi_stack)
    
    plot2_elnp = _PlotHelper(
        x=p_stack,
        y=e_stack,
        xlabel="ln $p$ [-]",
        ylabel="$e$ [-]",
        xlim=[0, None],
        ylim=[0, e_stack.max()*1.2],
        xlogscale=True
    )

    # Plot 3: M - e
    M_stack = q_stack/p_stack
    
    plot3_eM = _PlotHelper(
        x=M_stack,
        y=e_stack,
        xlabel="$M$ [-]",
        ylabel="$e$ [-]",
        xlim=[M_stack.min()*0.99, M_stack.max()*1.01],
        ylim=[0, e_stack.max()*1.2]
    )
    
    # Plot 4: deps_v_dt vs dgamma_dt
    deps_dt_stack = get_sym_tensor_stack(L_stack)
    dgamma_dt_stack = get_scalar_shear_strain_stack(deps_dt_stack)
    deps_v_dt_stack = get_volumetric_strain_stack(deps_dt_stack)
    
    plot4_deps_v_dt_dgamma_dt = _PlotHelper(
        x=deps_v_dt_stack,
        y=dgamma_dt_stack,
        xlabel="$\dot\\varepsilon_v$ [-]",
        ylabel="$\dot\\gamma$ [-]",
        xlim=[deps_v_dt_stack.min()*0.8, deps_v_dt_stack.max()*1.2],
        ylim=[dgamma_dt_stack.min()*0.8, dgamma_dt_stack.max()*1.2]
    )
    

    
    # Plot 5: log p - phi
    plot5_lnp_phi = _PlotHelper(
        x=phi_stack,
        y=p_stack,
        xlabel="$\phi$ [-]",
        ylabel="ln $p$ [-]",
        ylogscale=True,
        xlim=[phi_stack.min()*0.99, phi_stack.max()*1.01],
        ylim=[p_stack.min()*0.1,p_stack.max()*10], # adjust for logscale
    )
    
    # Plot 6: M - phi
    plot6_Mphi = _PlotHelper(
        y=M_stack,
        x=phi_stack,
        xlabel="$\phi$ [-]",
        ylabel="$M$ [-]",
        xlim=[phi_stack.min()*0.99, phi_stack.max()*1.01],
        ylim=[M_stack.min()*0.99, M_stack.max()*1.01],
    )
    
    fig_ax = make_plots(
        [
            plot1_qp,
            plot2_elnp,
            plot3_eM,
            plot4_deps_v_dt_dgamma_dt,
            plot5_lnp_phi,
            plot6_Mphi
            ],
        fig_ax = fig_ax
    )
    return fig_ax


def plot_set2(
    stress_stack: chex.Array,
    L_stack: chex.Array,
    F_stack: chex.Array,
    plot_helper_args: Dict = None,
    fig_ax: Tuple = None
    )-> Tuple:
    """Create the plot set 1.
    
    Plots included:
    
    q vs gamma | p vs gamma | M vs gamma
    q vs dot gamma | p vs dot gamma | M vs dot gamma

    Args:
        stress_stack (chex.Array): list of stress tensors
        L_stack (chex.Array): list of velocity gradients
        F_stack (chex.Array): list of deformation gradients
        fig_ax (Tuple, optional): fig axis pair. Defaults to None.

    Returns:
        Tuple: Update fig axes pair
    """
    
    # pass arguments to plot helper from outside
    if plot_helper_args is None:
        plot_helper_args={}
    
    _PlotHelper = partial(PlotHelper,**plot_helper_args)

    # Plot 1: q - gamma
    eps_stack,*_ = get_hencky_strain_stack(F_stack)
    
    
    gamma_stack = get_scalar_shear_strain_stack(eps_stack)
    
    q_stack = get_q_vm_stack(stress_stack)

    plot1_q_gamma= _PlotHelper(
        x=gamma_stack,
        y=q_stack,
        xlabel="$\gamma$ [-]",
        ylabel="$q$ [Pa]",
        xlim=[0, gamma_stack.max()*1.2],
        ylim=[0, q_stack.max()*1.2],
    )
    
    # Plot 2: p vs gamma
    p_stack = get_pressure_stack(stress_stack)
    
    plot2_p_gamma= _PlotHelper(
        x=gamma_stack,
        y=p_stack,
        xlabel="$\gamma$ [-]",
        ylabel="$p$ [Pa]",
        xlim=[0, gamma_stack.max()*1.2],
        ylim=[0, p_stack.max()*1.2]
    )
    
    
    # Plot 3: M vs gamma
    p_stack = get_pressure_stack(stress_stack)
    M_stack = q_stack/p_stack
    
    plot2_M_gamma= _PlotHelper(
        x=gamma_stack,
        y=M_stack,
        xlabel="$\gamma$ [-]",
        ylabel="$M$ [-]",
        xlim=[0, gamma_stack.max()*1.2],
        ylim=[M_stack.min()*0.99, M_stack.max()*1.01],
    )
    
    
    # Plot 4: q vs dot gamma
    q_stack = get_q_vm_stack(stress_stack)
    
    deps_dt_stack = get_sym_tensor_stack(L_stack)
    
    dgamma_dt_stack = get_scalar_shear_strain_stack(deps_dt_stack)
    
    plot4_q_dgamma_dt= _PlotHelper(
        x=dgamma_dt_stack,
        y=q_stack,
        xlabel="$\dot\gamma$ [-]",
        ylabel="$q$ [Pa]",
        xlim=[0, dgamma_dt_stack.max()*1.2],
        ylim=[0, q_stack.max()*1.2],
    )
    
    # Plot 5: p vs dot gamma
    plot5_p_dgamma_dt= _PlotHelper(
        x=dgamma_dt_stack,
        y=p_stack,
        xlabel="$\dot\gamma$ [-]",
        ylabel="$p$ [Pa]",
        xlim=[0, dgamma_dt_stack.max()*1.2],
        ylim=[0, p_stack.max()*1.2],
    )
    

    # Plot 6: M vs dot gamma

    
    plot6_M_dgamma_dt= _PlotHelper(
        x=dgamma_dt_stack,
        y=M_stack,
        xlabel="$\dot\gamma$ [-]",
        ylabel="$M$ [-]",
        xlim=[0, dgamma_dt_stack.max()*1.2],
        ylim=[M_stack.min()*0.99, M_stack.max()*1.01],
    )
    
    fig_ax = make_plots(
        [
            plot1_q_gamma,
            plot2_p_gamma, 
            plot2_M_gamma,
            plot4_q_dgamma_dt,
            plot5_p_dgamma_dt,
            plot6_M_dgamma_dt
            ],
        fig_ax = fig_ax
    )
    
    return fig_ax

def plot_set3(
    stress_stack: chex.Array,
    phi_stack: chex.Array,
    L_stack: chex.Array,
    F_stack: chex.Array,
    t_stack: chex.Array,
    plot_helper_args: Dict = None,
    fig_ax: Tuple = None
    ):
    """Create plot set 3:
    
    Plots include:
    q - t | p - t | M - t |
    phi - t | gamma -t | dgamma_dt - t

    Args:
        stress_stack (chex.Array): list of stress tensors
        phi_stack (chex.Array): list of solid volume fractions
        L_stack (chex.Array): list of velocity gradients
        F_stack (chex.Array): list of deformation gradients
        t_stack (chex.Array): time stack

    Returns:
        Typle: Updated fix axes pair
    """
    
    # pass arguments to plot helper from outside
    if plot_helper_args is None:
        plot_helper_args={}
    
    _PlotHelper = partial(PlotHelper,**plot_helper_args)
    
    # Plot 1: q - t

    q_stack = get_q_vm_stack(stress_stack)

    plot1_q_t= PlotHelper(
        x=t_stack,
        y=q_stack,
        xlabel="$t$ [s]",
        ylabel="$q$ [Pa]",
        ylim=[0, q_stack.max()*1.2],
    )
    
    # Plot 2: p - t

    p_stack = get_pressure_stack(stress_stack)
    
    plot2_p_t= _PlotHelper(
        x=t_stack,
        y=p_stack,
        xlabel="$t$ [s]",
        ylabel="$p$ [Pa]",
        ylim=[0, p_stack.max()*1.2],
    )
 
    # Plot 3: M - t
    M_stack = q_stack/p_stack
    plot3_M_t= _PlotHelper(
        x=t_stack,
        y=M_stack,
        xlabel="$t$ [s]",
        ylabel="$M$ [-]",
        ylim=[M_stack.min()*0.99, M_stack.max()*1.01],
    )

    
    # Plot 4: phi - t

    plot4_phi_t= _PlotHelper(
        x=t_stack,
        y=phi_stack,
        xlabel="$t$ [s]",
        ylabel="$\phi$ [-]",
        ylim=[phi_stack.min()*0.99, phi_stack.max()*1.01],
    )

    # Plot 5: gamma - t
    eps_stack,*_ = get_hencky_strain_stack(F_stack)
    gamma_stack = get_scalar_shear_strain_stack(eps_stack)

    plot5_gamma_t= _PlotHelper(
        x=t_stack,
        y=gamma_stack,
        xlabel="$t$ [s]",
        ylabel="$\gamma$ [-]",
        ylim=[gamma_stack.min()*0.9, gamma_stack.max()*1.1],
    )
    
    # Plot 6: dot gamma - t
        
    deps_dt_stack = get_sym_tensor_stack(L_stack)
    
    dgamma_dt_stack = get_scalar_shear_strain_stack(deps_dt_stack)
    
    plot6_dgamma_dt_t= _PlotHelper(
        x=t_stack,
        y=dgamma_dt_stack,
        xlabel="$t$ [s]",
        ylabel="$\dot\gamma$ [-]",
        ylim=[dgamma_dt_stack.min()*0.9, dgamma_dt_stack.max()*1.1],
    )
    
    fig_ax = make_plots(
        [
            plot1_q_t,
            plot2_p_t,
            plot3_M_t,
            plot4_phi_t,
            plot5_gamma_t,
            plot6_dgamma_dt_t
        ],
        fig_ax = fig_ax
    )
    
    return fig_ax
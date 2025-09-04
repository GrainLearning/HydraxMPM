# Copyright (c) 2024
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

import optimistix as optx
from typing_extensions import Optional, Self, Tuple

from ..common.types import (
    TypeFloat,
    TypeFloatScalarPStack,
    TypeFloatMatrixPStack,
    TypeInt,
)
from ..material_points.material_points import MaterialPoints
from ..utils.math_helpers import (
    get_dev_strain,
    get_dev_strain_stack,
    get_dev_stress,
    get_pressure,
    get_q_vm,
    get_volumetric_strain,
    get_inertial_number,
    get_scalar_shear_strain,
    get_trx_shear_strain
)
from .constitutive_law import ConstitutiveLaw, ConvergenceControlConfig

def get_p(p_prev, p_s, deps_e_v, kap):
    """Compute non-linear pressure."""
    p_next  = (p_prev + p_s)*jnp.exp(deps_e_v / kap) - p_s
    return jnp.clip(p_next, 1.0, None)


def get_p_c(p_c_prev, ps, deps_p_v, lam, kap, I, I_v):
    """Compute non-linear pressure."""
    p_c_next  = (p_c_prev + ps)*jnp.exp((deps_p_v +I/I_v)/ (lam - kap)) - ps
    return jnp.clip(p_c_next, 1.0, None)



def get_K(kap, p, ps, K_min=None, K_max=None):
    p_ = jnp.clip(p+ps, 1.0, None)
    K = (p_ / kap) 
    K = jnp.clip(K, K_min, K_max)
    return K

def get_G(nu, K):
    G = (3 * (1 - 2 * nu) / (2 * (1 + nu))) * K
    return G

def yield_function_mcc(p, p_c, q, M):
    """Yield function for modified Cam Clay model."""  
    eta = q / p
    
    return eta**2 -(M**2)*(p_c/p -1)

def get_s(deps_e_d, G, s_prev):
    return 2.0 * G * deps_e_d + s_prev



def v_swelling_line(v_0, p,p_c,lam,kap,p_s=0.0,I=0.0, I_v=20_00):
    v_lam_ = v_lambda(p_c, p_s, lam)
    v_kap_ = v_kappa(p, p_c, p_s, kap)
    v_I_ = v_I(I, I_v)
    return v_0*v_lam_ * v_kap_ * v_I_


def v_lambda(p_c, p_s, lam):

    return ((p_c + p_s)/p_s)**(-lam)

def v_kappa(p, p_c, p_s, kap):

    return ((p+p_s)/(p_c + p_s))**(-kap)

def v_I(I,I_v):
    return jnp.exp(I/I_v)

def eta_f(p, p_c, gamma, beta):
    return 1.0- beta + beta*gamma*(1/2)*(p_c/p)

def eta_h(p, p_c, gamma):
    p_c_ = gamma*(1/2)*(p_c/p) 
    return jnp.sqrt(1-((1-p_c_)/ (1.0 - gamma + p_c_))**2 )



def eta_ys(M,q,p,eta_f_2,eta_h_2,eta_I_2):
 return (q/(p*M))**2 - eta_f_2 * eta_h_2* eta_I_2


def eta_f_2(p, p_c, gamma, beta):
    return (1.0- beta + beta*gamma*(1/2)*(p_c/p))**2

def eta_h_2(p, p_c, gamma):
    p_c_ = gamma*(1/2)*(p_c/p) 
    return 1-((1-p_c_)/ (1.0 - gamma + p_c_))**2 

def eta_I_2(I, I_0, M, M_d):
    return (1 + I * ((M_d/M - 1.0) / (I_0 + I)))**2



class IsotropicFamilySoilsInertia(ConstitutiveLaw):
    nu: TypeFloat
    M: TypeFloat
    lam: TypeFloat
    kap: TypeFloat
    N: Optional[TypeFloat] = None
    N_star: Optional[TypeFloat] = None
    v_0: Optional[TypeFloat] = None
    p_s: Optional[TypeFloat] = None
    p_ref: TypeFloat = 1000.0  # Reference pressure in Pa (1 kPa)

    gamma: TypeFloat = 1
    beta: TypeFloat = 1
    r: TypeFloat = 2.0
    OCR: TypeFloat
    
    I_v: TypeFloat = 20_00
    I_0: TypeFloat = 20_00
    M_d: TypeFloat = 1_00
    
    # internal state variables
    p_c_stack: Optional[TypeFloatScalarPStack] = None
    I_stack: Optional[TypeFloatScalarPStack] = None
    
    # initial stress
    stress_0_stack: Optional[TypeFloatMatrixPStack] = None

    settings: ConvergenceControlConfig
    
    # Needed? 
    K_min: Optional[TypeFloat] = None
    K_max: Optional[TypeFloat] = None


    def __init__(
        self: Self,
        nu: TypeFloat,
        M: TypeFloat,
        OCR: TypeFloat,
        lam: TypeFloat,
        kap: TypeFloat,
        K_min: TypeFloat = None,
        K_max: TypeFloat = None,
        r= 2.0,
        beta: TypeFloat = 1.0,
        p_ref: TypeFloat = 1000.0,  # Reference pressure in Pa (1 kPa)
        N: Optional[TypeFloat] = None,
        N_star: Optional[TypeFloat] = None,
        v_0: Optional[TypeFloat] = None,
        p_s: Optional[TypeFloat] = None,
        I_0: Optional[TypeFloat] = None,
        I_v: Optional[TypeFloat] = None,
        M_d: Optional[TypeFloat] = None,
        settings: Optional[dict | ConvergenceControlConfig] = None,
        **kwargs,
    ) -> Self:
        """_summary_

        Args:
            self (Self): _instance of the class
            nu (TypeFloat): Poison's ratio
            M (TypeFloat): critical state ratio
            OCR (TypeFloat): overconsolidation ratio
            lam (TypeFloat): slope of normal consolidation line in double log space (at high pressures)
            kap (TypeFloat): slope of swelling line in double log space (at low pressures)
            r (float, optional): spacing ratio p_c/pcsl. Defaults to 2.0.
            beta (TypeFloat, optional): shear distortion parameter. Defaults to 1.0.
            p_ref (TypeFloat, optional): reference pressure where N is taken. Defaults to 1000.0.
            v_0 (Optional[TypeFloat], optional): virtual stress-free specific volume. Defaults to None.
            p_s (Optional[TypeFloat], optional): compressive hardening parameter. Defaults to None.
            N_star (Optional[TypeFloat], optional): curvature parameter. Defaults to None.
            settings (Optional[dict  |  ConvergenceControlConfig], optional): _description_. Defaults to None.
            K_min (TypeFloat, optional): minimum bulk modulus. Defaults to None.
            K_max (TypeFloat, optional): maximum bulk modulus. Defaults to None.
            I_0 (Optional[TypeFloat], optional): characteristic inertial number for mu(I) rheology. Defaults to None.
            I_v (Optional[TypeFloat], optional): characteristic inertial number for phi(I) rheology. Defaults to None.
            M_d (Optional[TypeFloat], optional): inertial steady state ratio.

        Returns:
            Self: instance of self
        """
        
        # inertial parameters
        # default values are set to large numbers for rate independent behaviour
        if I_v is None:
            I_v = 1e20
        self.I_v = I_v
    
        if I_0 is None:
            I_0 = 1e20
        self.I_0 = I_0 

        if M_d is None:
            M_d = M
        self.M_d = M_d
        
        
        # soil model parameters
        self.r = r
        self.gamma = 2/r
        
        self.beta = beta
        self.nu = nu

        self.M = M

        self.OCR = OCR

        self.lam = lam

        self.kap = kap


        self.N = N

        self.K_min = K_min

        self.K_max = K_max


        rho_0 = kwargs.get("rho_0", None) # initial density
        rho_p = kwargs.get("rho_p", None) # particle density
        
        self.p_ref = p_ref

        if p_s == 0.0:
            self.p_s = 0
            self.N = N
            self.v_0 = N
        elif v_0 is not None:
            self.v_0 = v_0
            self.p_s = p_ref*(v_0/N)**self.lam
            self.N_star = N*(self.p_s/self.p_ref +1)**(1/self.lam)
            self.rho_0 = rho_p / v_0
            kwargs["rho_0"] = self.rho_0
        elif N_star is not None:
            self.N_star = N_star
            self.p_s = ((N_star/N)**self.lam -1)
            self.v_0 = (self.p_s/p_ref)**(1/self.lam) * N
            self.rho_0 = rho_p / self.v_0
            kwargs["rho_0"] = self.rho_0
        elif rho_0 is not None:
            self.v_0 = rho_p / rho_0
            self.p_s = p_ref*(self.v_0/N)**self.lam
            self.N_star = N*(self.p_s/self.p_ref +1)**(1/self.lam)


        self.eps_e_stack = kwargs.get("eps_e_stack")

        self.p_c_stack = kwargs.get("p_c_stack")

        self.stress_0_stack = kwargs.get("stress_0_stack")
        self.I_stack = kwargs.get("I_stack")

        # settings used for convergence control
        if settings is None:
            settings = dict()
        if isinstance(settings, dict):
            self.settings = ConvergenceControlConfig(
                rtol=settings.get("rtol", 1e-4),
                atol=settings.get("atol", 1e-4),
                max_iter=settings.get("max_iter", 40),
                throw=settings.get("throw", True),
                # plastic multiplier and volumetric strain, respectively
                lower_bound=settings.get("lower_bound", (0.0, -10.0, -10.0)),
            )
        else:
            self.settings = settings
        del settings

        super().__init__(**kwargs)

    def init_state(self: Self, material_points: MaterialPoints):
        stress_0_stack = material_points.stress_stack
        p_0_stack = material_points.p_stack
        p_c_stack = p_0_stack * self.OCR

        
        v = jax.vmap(v_swelling_line, in_axes=(None,0,0,None,None,None))(
            self.v_0, p_0_stack, p_c_stack, self.lam, self.kap, self.p_s
        )
        # bulk density
        rho = self.rho_p / v

        eps_e_stack = jnp.zeros((material_points.num_points, 3, 3))
        
        
        I_stack = jnp.zeros(material_points.num_points)
        return self.post_init_state(
            material_points,
            rho_0=self.rho_0,
            rho=rho,
            stress_0_stack=stress_0_stack,
            p_c_stack=p_c_stack,
            eps_e_stack=eps_e_stack,
            I_stack=I_stack,
        )

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """Update the material state and particle stresses for MPM solver."""

        deps_dt_stack = material_points.deps_dt_stack

        new_stress_stack, new_eps_e_stack, new_p_c_stack,new_I_stack = (
            self.vmap_constitutive_update(
                dt,
                deps_dt_stack,
                self.eps_e_stack,
                material_points.stress_stack,
                self.p_c_stack,
                self.stress_0_stack,
                material_points.specific_volume_stack(self.rho_p),
                material_points.isactive_stack,
                self.I_stack,

            )
        )
        new_self = eqx.tree_at(
            lambda state: (state.eps_e_stack, state.p_c_stack,state.I_stack),
            self,
            (new_eps_e_stack, new_p_c_stack, new_I_stack),
        )

        new_material_points = eqx.tree_at(
            lambda state: (state.stress_stack),
            material_points,
            (new_stress_stack),
        )
   
        return new_material_points, new_self

    @partial(
        jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0,0), out_axes=(0, 0, 0,0)
    )
    def vmap_constitutive_update(
        self: Self,
        dt,
        deps_dt_next,
        eps_e_prev,
        stress_prev,
        p_c_prev,
        stress_0,
        specific_volume,
        isactive,
        I_prev,
    ):
        default_values = (stress_prev, eps_e_prev, p_c_prev, I_prev)

        def update(_):
            return self.update_ip(
                dt,
                deps_dt_next,
                eps_e_prev,
                stress_prev,
                p_c_prev,
                stress_0,
                specific_volume,
                I_prev,
         
            )

        return jax.lax.cond(
            isactive,
            update,
            lambda _: default_values,
            operand=None,  # No additional operand needed
        )

    def update_ip(
        self: Self,
        dt,
        deps_dt_next,
        eps_e_prev,
        stress_prev,
        p_c_prev,
        stress_0,
        specific_volume,
        I_prev,

    ):
        deps_next = deps_dt_next * dt
        
        # reference stresses
        p_0 = get_pressure(stress_0)
        s_0 = get_dev_stress(stress_0, pressure=p_0)

        # previous stresses
        p_prev = get_pressure(stress_prev)
        s_prev = get_dev_stress(stress_prev, pressure=p_prev)

        # trail elastic volumetric strain
        deps_e_v_tr = get_volumetric_strain(deps_next)

        # trail pressure in transformed space
        p_tr = get_p(p_prev, self.p_s, deps_e_v_tr, self.kap)

        # trail bulk and shear modulus in transformed space
        K_tr = get_K(self.kap, p_tr, self.p_s,self.K_min, self.K_max)
        G_tr = get_G(self.nu, K_tr)

        # trail elastic deviatoric strain tensor
        deps_e_d_tr = get_dev_strain(deps_next, deps_e_v_tr)
        

        # trail elastic deviatoric stress tensor
        s_tr = get_s(deps_e_d_tr, G_tr, s_prev)

        # trail von Mises stress
        q_tr = get_q_vm(dev_stress=s_tr)

        
        eta_f_2_tr = eta_f_2(p_tr, p_c_prev, self.gamma, self.beta)
        eta_h_2_tr = eta_h_2(p_tr, p_c_prev, self.gamma)
        
        # yield surface
        yf = eta_ys(self.M, q_tr, p_tr, eta_f_2_tr, eta_h_2_tr,1.0)
        
        is_ep = yf > 0.0


        def elastic_update():
            stress_next = s_tr - p_tr * jnp.eye(3)
            eps_e_tr = eps_e_prev + deps_next
            return stress_next, eps_e_tr, p_c_prev, 0.0

        def pull_to_ys():
            # https://github.com/patrick-kidger/optimistix/issues/132
            # prevents non-sensical updates when tracing the return mapping algorithm
            # since jax evaluates both branches of the cond statement
            deps_e_v_tr_ = jnp.where(~is_ep, 0.0, deps_e_v_tr)
            p_prev_ = jnp.where(~is_ep, p_0 + self.p_s, p_prev)
            p_c_prev_ = jnp.where(~is_ep, p_0 + self.p_s, p_c_prev)
            q_tr_ = jnp.where(~is_ep, 0.0, q_tr)



            def residuals(sol, args):
                pmulti, deps_p_v, dI_next = sol
                is_ep_ = is_ep
            
                deps_e_v = deps_e_v_tr_ - deps_p_v
                
                p_next = get_p(p_prev_, self.p_s,deps_e_v, self.kap)
             
                K_next = get_K(self.kap, p_next, self.p_s,self.K_min, self.K_max)
    
                G_next = get_G(self.nu, K_next)
                

                # next deviatoric strain tensor end Von Mises stress tensor
                factor = (1 / (1 + 6.0 * G_next * pmulti))

                s_next = s_tr * factor
                q_next = q_tr_ * factor
                
                # begin inertial part
                deps_s_p_dt = (2 * pmulti * q_next) / dt
                
                I_next_fr = get_inertial_number(
                    jnp.clip(p_next, 1, None),
                    jnp.sqrt(2/3)*deps_s_p_dt,
                    self.d,
                    self.rho_p,
                )
                
                dI_next_fr = I_next_fr-I_prev
    
                p_c_next = get_p_c(p_c_prev_, self.p_s, deps_p_v, self.lam, self.kap,
                                   dI_next,
                                   self.I_v)

                
                eta_I_2_next = eta_I_2(I_prev+dI_next, self.I_0, self.M, self.M_d)
                eta_f_2_next = eta_f_2(p_next, p_c_next, self.gamma, self.beta)
                eta_h_2_next = eta_h_2(p_next, p_c_next, self.gamma)
                
                
                A = (1-self.gamma)*p_next +(1/2)*self.gamma*p_c_next
                B = self.M*((1-self.beta)*p_next +(1/2)*self.gamma*self.beta*p_c_next)

                deps_v_p_fr = (
                    pmulti
                    *(2*p_next - self.gamma*p_c_next)
                    *(B/A)**2
                ) 

      
                
                yf_next = eta_ys(self.M, q_next, p_next, eta_f_2_next, eta_h_2_next,
                                 eta_I_2_next
                                 ) 


                R = jnp.array(
                    [
                        
                    yf_next,
                     (deps_v_p_fr - deps_p_v),
                    (dI_next - dI_next_fr)/self.I_v,

                     ]
                )

                aux = (
                    p_next,
                    s_next,
                    p_c_next,
                    G_next,
                    K_next
                )
                return R, aux

            def find_roots():
                """Find roots of the residuals function."""

                init_val = jnp.array([0.0,0.0,0.0])

                solver = optx.Newton(
                    rtol=self.settings.rtol, atol=self.settings.atol, 
                    norm=optx.rms_norm
                )

                sol = optx.root_find(
                    residuals,
                    solver,
                    init_val,
                    throw=self.settings.throw,
                    has_aux=True,
                    max_steps=self.settings.max_iter,
                    options=dict(
                        lower=jnp.array(self.settings.lower_bound),
                    ),
                )
                return sol.value

            pmulti_curr, deps_p_v_next,dI_next = find_roots()

            R, aux = residuals([pmulti_curr, deps_p_v_next,dI_next], None)
            


            p_next, s_next, p_c_next, G_next, K_next = aux

            
            I_next = I_prev + dI_next
            
            stress_next = s_next - p_next * jnp.eye(3)

            eps_e_v_next = (p_next - p_0) / K_next

            eps_e_d_next = (s_next - s_0) / (2.0 * G_next)

            eps_e_next = eps_e_d_next - (1.0 / 3) * eps_e_v_next * jnp.eye(3)

            return stress_next, eps_e_next, p_c_next,I_next


        # not sure how to handle the case at low pressures,
        # since material points are generally considered stress free
        # if they are above a certain specific volume
        stress_next, eps_e_next, p_c_next,I_next = jax.lax.cond(
            specific_volume <= self.v_0,
            lambda: jax.lax.cond(is_ep, pull_to_ys, elastic_update),
            lambda: jax.lax.cond(is_ep, pull_to_ys, elastic_update),
            # lambda: (0.0 * jnp.eye(3), eps_e_prev, p_c_prev, 0.0), 
        )

        return stress_next, eps_e_next, p_c_next, I_next

    def NCL_specific_volume(self, p_stack,I=0.0):
        """Equation for critical state line (CSL) in double log specific volume/pressure space (ln v - ln p) space.

        Used for plotting purposes only.
        
        Returns specific volume (not logaritm)
        """
        p_c_stack = p_stack


        return jax.vmap(v_swelling_line, in_axes=(None,0,0,None,None,None,None,None))(
            self.v_0, p_stack, p_c_stack, self.lam, self.kap, self.p_s, I, self.I_v
        )

    def CSL_specific_volume(self, p_stack,I=0.0):
        """Equation for critical state line (CSL) in double log specific volume/pressure space (ln v - ln p) space.

        Used for plotting purposes only.

        Returns specific volume (not logaritm)
        """
        p_c_stack = p_stack*self.r


        return jax.vmap(v_swelling_line, in_axes=(None,0,0,None,None,None,None,None))(
            self.v_0, p_stack, p_c_stack, self.lam, self.kap, self.p_s, I, self.I_v
        )

    def CSL_q(self, p_stack):
        """Equation for critical state line (CSL) in scalar shear stress- pressure (q - p) space.

        Used for plotting purposes only.
        
        Returns specific volume (not logaritm)
        """
        return p_stack * self.M

    # def get_dt_crit(self, material_points, cell_size, dt_alpha=0.5):

    #     """Get critical timestep of material poiints for stability."""

    #     def vmap_dt_crit(p, rho, vel):
    #         K = get_K(self.kap, p, self.K_min, self.K_max)
    #         G = get_G(self.nu, K)

    #         cdil = jnp.sqrt((K + (4 / 3) * G) / rho)

    #         c = jnp.abs(vel) + cdil * jnp.ones_like(vel)
    #         return c

    #     c_stack = jax.vmap(vmap_dt_crit)(
    #         material_points.p_stack,
    #         material_points.rho_stack,
    #         material_points.velocity_stack,
    #     )
    #     return (dt_alpha * cell_size) / jnp.max(c_stack)

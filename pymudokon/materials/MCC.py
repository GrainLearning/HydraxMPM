"""
Contains the Modified Cam Clay constitutive model.

The model is designed as a standalone for educational and research purposes.
It follows the return-mapping procedure described in the book:

    "Computational Methods for Plasticity: Theory and Applications"
    by Eduardo A. de Souza Neto, Djordje Peric, and David R.J. Owen
    Published by John Wiley & Sons, 2011

There are a few exceptions to the procedure described in the book:
- Compression is considered positive.
- The hardening rule follows the traditional e-ln p formulation.

The elasticity of the model is based on linear isotropic elasticity,
which is valid for small pressure ranges.
"""
import numpy as np


def compute_yield_function(p: float, ps: float, q: float, M: float)->float:
    """
    Compute the modified Cam Clay yield function.

    Args:
    ----
        p (float): pressure
        ps (float): backstress
        q (float): von Mises stress
        M (float): slope of the critical state line

    Returns:
    -------
        float: yield function value

    """
    return (ps - p) ** 2 + (q / M) ** 2 - ps**2


class ModifiedCamClay:
    def __init__(
        self: "ModifiedCamClay",
        E: float,
        nu: float,
        M: float,
        lam: float,
        kap: float,
        Vs: float,
        R: float,
        volume: float,
        reference_stress: np.ndarray,
        reference_strain: np.ndarray = None,
    ) -> None:
        """
        Initialize the MCC (Modified Cam Clay) model parameters.

        Parameters
        ----------
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        M : float
            Slope of the critical state line.
        lam : float
            Compression index.
        kap : float
            Decompression index.
        Vs : float
            Volumetric strain.
        R : float
            Over consolidation ratio.
        dt : float
            Time step.
        volume : float
            Initial volume.
        reference_stress : np.ndarray
            The pre-stress.
        reference_strain : np.ndarray, optional
            The pre-strain. Defaults to None.
        isLinearElast : bool, optional
            If True, the model is consistent. Defaults to True.

        """
        #### ELASTICIY ####
        # Set Young's modulus (E), Poisson's ratio (nu),
        # Bulk modulus (K), and shear modulus (G)
        self.E = E
        self.nu = nu
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.G = E / (2.0 * (1.0 + nu))

        ##### YIELD SURFACE SHAPE ####
        # Set Slope of critical state line (M),
        # Over consolidation ratio (R)
        self.M = M
        self.R = R

        ##### HARDENING PARAMETERS ####
        # Compression slope (lam), Decompression slope (kap)
        # Current volumetric strain (Vs)
        self.lam = lam
        self.kap = kap
        self.Vs = Vs

        #### REFERENCE STATE ####
        # Reference stress, pressure and deviatoric stress
        self.stress_ref = reference_stress
        self.p_ref = -np.trace(reference_stress) / 3.0
        self.s_ref = self.stress_ref + self.p_ref * np.eye(3)

        # Reference strain
        if reference_strain is None:
            self.strain_ref = np.zeros((3, 3))

        #### INTERNAL STATE ####
        # Back stress (pc), plastic volumetric strain (eps_v_p),
        # Elastic strain (eps_e)
        self.pc = self.p_ref * R
        self.eps_v_p = 0.0
        self.eps_e = np.zeros(3)

        self.volume = volume

        self.stress = np.zeros((3, 3))

    def stress_update(
        self: "ModifiedCamClay",
        strain_rate: np.ndarray,
        dt: float,
        update_history: bool = True,
    ) -> None:
        """
        Perform a stress update step of the modified Cam Clay model.

        Args:
        ----
            self (ModifiedCamClay): self reference
            strain_increment (np.ndarray): 3x3 strain rate tensor
            dt (float): timestep
            update_history (bool, optional): Flag if history should be updated.
            Used in stress controlled boundary conditions. Defaults to True.

        """
        # aka as volumetric strain, take compression is positive
        self.volume_rate_change = -np.trace(strain_rate)

        self.volume = self.volume * (1 - self.volume_rate_change)

        strain_increment = strain_rate * dt

        #### ELASTIC PREDICTOR STEP ####
        eps_e_trail = self.strain_ref + self.eps_e + strain_increment

        # Get trail elastic volumetric elastic strain, deviatoric elastic strain
        eps_v_e_trail = -np.trace(eps_e_trail)
        eps_d_e_trail = eps_e_trail + (1.0 / 3) * eps_v_e_trail * np.eye(3)

        # Get trail pressure, deviatoric stress, von Mises stress
        p_trail = self.p_ref + self.K * eps_v_e_trail
        s_trail = self.s_ref + 2.0 * self.G * eps_d_e_trail
        q_trail = np.sqrt(1.5 * (s_trail @ s_trail.T).trace())

        # Get back stress
        ps_trail = 0.5 * self.pc

        # Compute yield function
        f_trail = compute_yield_function(p_trail, ps_trail, q_trail, self.M)

        # If within yeild surface, return stress
        if f_trail <= 0.0:
            if update_history:
                self.eps_e = eps_e_trail.copy()

            self.stress = s_trail - p_trail * np.eye(3)

            return
        # yield surface is crossed, perform return mapping

        specific_volume = self.volume / self.Vs

        v_lam_tilde = specific_volume / (self.lam - self.kap)

        #### RETURN MAPPING ####
        # initial guess for plastic multiplier and plastic volumetric strain
        Solution = np.array([0.0, self.eps_v_p])
        tol = 1e-2

        R = np.zeros(2)

        while True:
            pmultp, eps_p_v_next = Solution  # unpack solution

            # compute plastic volumetric strain increment
            deps_p_v = eps_p_v_next - self.eps_v_p

            # compute updated pressure, von Mises stress, back stress, deviatoric stress
            p_next = p_trail - self.K * deps_p_v

            q_next = (self.M**2 / (self.M**2 + 6.0 * self.G * pmultp)) * q_trail

            ps_next = 0.5 * (self.pc * (1.0 + v_lam_tilde * deps_p_v))

            s_next = (self.M**2 / (self.M**2 + 6.0 * self.G * pmultp)) * s_trail

            # compute residuals
            R[0] = compute_yield_function(p_next, ps_next, q_next, self.M)
            R[1] = eps_p_v_next - self.eps_v_p + 2.0 * pmultp * (ps_next - p_next)

            # hardening slope
            H = 0.5 * v_lam_tilde * (self.pc / (1.0 - v_lam_tilde * deps_p_v) ** 2)

            # difference between yield surface radius and pressure
            p_overline = ps_next - p_next

            Jac = np.zeros((2, 2))

            Jac[0, 0] = ((-12.0 * self.G) / (self.M**2 + 6.0 * self.G * pmultp)) * (
                q_next / self.M
            ) ** 2

            Jac[0, 1] = (2.0 * p_overline) * (self.K + H) - 2.0 * ps_next * H

            Jac[1, 0] = 2.0 * p_overline

            Jac[1, 1] = 1.0 + (2.0 * pmultp) * (self.K + H)

            inv_Jac = np.linalg.inv(Jac)

            Solution = Solution - inv_Jac @ R

            # normalize to magnitude similar to other residuals (e.g. R[1] strains)
            R[0] /= self.K
            conv = np.linalg.norm(R)

            if abs(conv) < tol:
                break

        self.stress = s_next - p_next * np.eye(3)
        if update_history:
            self.eps_e = (s_next - self.s_ref) / (2.0 * self.G) - (
                p_next - self.p_ref
            ) / (3.0 * self.K) * np.eye(3)

            self.eps_v_p = eps_p_v_next
            self.pc = 2.0 * ps_next

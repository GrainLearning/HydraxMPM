import jax
import jax.numpy as jnp
import equinox as eqx
import hydraxmpm as hdx

from UEPI_MCC import UEPI_MCC

###########################################################################
# Loading Parameters & Configuration
###########################################################################

class SIPConfig(eqx.Module):

    # MCC parameters
    lam: float = 0.0925 # slope of csl [-]
    kap: float = 0.05 # slope of swelling line [-]
    N: float = 3.2535 # reference specific volume NCL [-]
    nu: float = 0.25 # Poisson's ratio [-]
    M_csl: float = 0.89 # critical state friction coefficient [-]

    # inertial steady state parameters
    M_inf: float = 1.35 # dynamic friction coefficient at high inertial number [-]
    I_v: float = 0.29e-9
    I_M: float = 1.0e-11

    # material characteristics
    d: float = 4e-6
    rho_p: float = 2700.0  # 2.7 * 1000

    # operational parameters
    OCR: float = 1.0
    confine: float = 50_000.0  # 50 kPa
    total_time: float = 3600.0 # 1 hour in seconds

    # label index for plotting
    label_idx: int = 0

    @property
    def gamma(self):
        return self.N * 2** -(self.lam - self.kap)

###########################################################################
# Loading procedure for numerical element simulations
###########################################################################

class NumericalProcedure(eqx.Module):
    num_steps: int = 2000
    total_strain: float = 0.5
    is_undrained: bool = False

    def __call__(self, config: SIPConfig):
        """
        Runs ONE simulation based on the provided config.
        Returns a dictionary or PyTree of results (Trajectories).
        """

        law = UEPI_MCC(
            nu=config.nu,
            M_csl=config.M_csl,
            M_inf=config.M_inf,
            lam=config.lam,
            kap=config.kap,
            N=config.N,
            p_ref=1000.0,
            d=config.d,
            rho_p=config.rho_p,
            I_v=config.I_v,
            I_M=config.I_M,
        )

        driver = hdx.ElementTestDriver(law)

        p_target = jnp.array([config.confine])
        law_state, stress_stack, density_stack = law.create_state_from_ocr(
            p_stack=p_target, ocr_stack=config.OCR
        )

        mp_state = hdx.MaterialPointState.create(
            stress_stack=stress_stack, density_stack=density_stack
        )

        dt = config.total_time / self.num_steps
        axial_rate = (self.total_strain / self.num_steps) / dt



        triaxial_test = hdx.TriaxialTest(
            solver=driver,
            confine=config.confine,
            is_undrained=self.is_undrained,
            axial_rate=jnp.asarray(axial_rate),
            num_steps=self.num_steps,
            dt=dt,
            stride=1,
            start_static=True,
        )

        axial_rate_hr = axial_rate * 100.0 * 3600.0

        jax.debug.print(
            "Running simulation with axial strain rate {:.2f} [%/hr] for {} hrs",
            axial_rate_hr,
            config.total_time / 3600.0,
        )
        mp_traj, law_traj = triaxial_test.run(mp_init=mp_state, law_init=law_state)

        return mp_traj, law_traj, triaxial_test.axial_strain_stack, axial_rate_hr

###########################################################################
# Loading procedure for numerical element simulations
###########################################################################

class LondonClayProcedure(eqx.Module):

    # Constant test settings (do not vary per batch)
    num_steps: int = 2000
    total_time: float = 3600.0
    is_undrained: bool = True
    

    def __call__(self, config: SIPConfig, exp_rates, exp_dt):

        law = UEPI_MCC(
            nu=config.nu,
            M_csl=config.M_csl,
            M_inf=config.M_inf,
            lam=config.lam,
            kap=config.kap,
            N=config.N,
            p_ref=1000.0,
            d=config.d,
            rho_p=config.rho_p,
            I_v=config.I_v,
            I_M=config.I_M,
        )

        driver = hdx.ElementTestDriver(law)

        p_target = jnp.array([config.confine])

        law_state, stress_stack, density_stack = law.create_state_from_ocr(
            p_stack=p_target, ocr_stack=config.OCR
        )


        mp_state = hdx.MaterialPointState.create(
            stress_stack=stress_stack, density_stack=density_stack
        )

        triaxial_test = hdx.TriaxialTest(
            solver=driver,
            confine=config.confine,
            is_undrained=self.is_undrained,
            axial_rate=exp_rates,
            num_newton_iters=20,
            dt=exp_dt,
            stride=1,
        )

        mp_traj, law_traj = triaxial_test.run(mp_init=mp_state, law_init=law_state)


        return mp_traj, law_traj, triaxial_test.axial_strain_stack
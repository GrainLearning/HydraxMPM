# Unified Elasto-Plastic-Inertial Modified Cam Clay (UEPI-MCC) constitutive law.

## Introduction

The following project contains implementation of the **Unified Elasto-Plastic-Inertial Modified Cam Clay (UEPI-MCC)** constitutive law [1].

This model combines hyperplastic Modified Cam Clay and the μ(I)–ϕ(I) rheology by multiplicative merging of state variables [1]. At the rate-independent limit, it reduces to the Modified Cam Clay model, which is a subset of the more general class of thermomechanical models [2]. While at the steady state limit it reduces to an μ(I)–ϕ(I) rheology, formulated as an inertial steady state [3, 4]. 

> **Note:** This model is compatible with the single point element test driver in hydraxmpm, but has not yet been publicly released for MPM simulations.

## Model Parameters

Parameters are inherited from Modified Cam Clay (MCC) and the Inertial Steady State (ISS) and listed in the following table.

| Parameter | Symbol | Category |
| :--- | :--- | :--- |
| **Critical state bulk friction** | $M_\mathrm{csl}$ | MCC / ISS |
| **Slope of Critical State Line** | $\lambda$ | MCC |
| **Slope of Swelling Line** | $\kappa$ | MCC |
| **Poisson’s ratio** | $\nu$ | MCC |
| **Reference critical state specific volume** | $\Gamma$ | MCC |
| **Maximum dynamic bulk friction** | $M_\infty$ | ISS |
| **Dilation characteristic inertial number** | $I_v$ | ISS |
| **Stress ratio characteristic inertial number** | $I_M$ | ISS |



## Project structure

The model is implemented within the HydraxMPM code. The implementation is done within the JAX-ecosystem.


### Core Implementation
- **`UEPI_MCC.py`**: implementation of the constitutive model

**Implementation details**

The implementation follows an incremental formulation and the return-mapping procedure is implicit in plasticity. The return-mapping procedure follows the generalized J2 style return-mapping similar to de Souza Neto [5], where the plastic deviatoric strain is phrased as only a function of deviatoric stress. This reduces the number of unknowns substantially. We use the Newton-Raphson method with automatic differentiation.


### Visuals
- **`theory_ys_plot.py`** plot of MCC yield surface in (p, q)
- **`theory_ys_sl.py`** plot of MCC swelling line in bilogarithmic (p, v)
- **`theory_iss.py`** plot of Inertial steady state in

### Benchmarks & results
- **`sip_setup.py`**: material parameters, loading procedures of the triaxial element tests
- **`results_trx.py`** triaxial undrained and drained element tests simulations showing effects of varying strain rates, ocr, and ISS parameters. Plotted are dashboard pressure--deviatoric shear $(p,q)$; bilogarithmic pressure--specific volume $(p, v)$; deviatoric strain--deviatoric shear ($\varepsilon_q, q$); deviatoric strain--pressure $(\varepsilon_q, p)$ for undrained or deviatoric strain--volumetric strain ($\varepsilon_q, p$).
- **`results_london_clay_trx.py`** Triaxial undrained element test of step-wise rate changes for reconstituted london clay [6]. 
- **`results_london_clay_iso.py`** Match NCL to isotropic compression of reconstituted london clay [6]

### Helpers
- **`plotting.py`** Contains plotting helper functions and matplotlib theme.


> **Note:** Experimental data for reconstituted London Clay is not included in this repository due to data ownership and licensing restrictions.

## References
- [1] Lubbe R., Cheng H, Luding S, Magnanimo V. Unified constitutive model bridging Critical State Soil  Mechanics and μ(I)–ϕ(I) Rheology, Submitted to Journal of the Mechanics and Physics of Solids, 2026.
- [2] Collins, I. F., B. Muhunthan, and B. Qu. "Thermomechanical state parameter models for sands." Géotechnique 60.8 (2010): 611-622.
- [3] GDR MiDi http://www. lmgc. univ-montp2. fr/MIDI/gdrmidi@ polytech. univ-mrs. fr. "On dense granular flows." The European Physical Journal E 14.4 (2004): 341-365.
- [4] Luding, Stefan, et al. "From particles in steady state shear bands via micro-macro to macroscopic rheology laws." International Conference on Discrete Element Methods. Singapore: Springer Singapore, 2016.
- [5] de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen. Computational methods for plasticity: theory and applications. John Wiley & Sons, 2008.
- [6] Sorensen, Kenny K., B. A. Baudet, and B. Simpson. "Influence of structure on the time-dependent behaviour of a stiff sedimentary clay." Géotechnique 57.1 (2007): 113-124.

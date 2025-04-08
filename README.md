<br /><br />

<p align="center">
  <img src="docs/_static/hydraxmpm.png" alt="HydraxMPM Logo" width="200">
</p>
<h1 align="center"><b>HydraxMPM</b></h1>
<p align="center">
  <b>A JAX-powered Material Point Method & Single Integration Point simulation environment for granular models</b>
<p align="center">

<!-- Integrated MPM & SIP tests
leads to shared API and rapid prototyping of granular constitutive laws

 Material Point Method solver developed with [JAX](https://jax.readthedocs.io/en/latest/quickstart.html)

This code is still in pre-alpha, i.e., all models may not be feature-complete. -->

  

## What Can You Do with HydraxMPM?
* 📈 Explore the behavior of granular materials under controlled conditions.
* ⛰️ Simulate complex events like landslides
* ⏳ Compare and validate different solid-like and fluid-like granular constitutive models.
* 🔬 Develop a deeper understanding of granular dynamics at both local and global levels.

## 🧠 Key Benefits

* **Unify local and global views:** Test, compare and validate constitutive models at a SIP-level, and directly apply them to large-scale MPM simulations – all within the same framework. 
* **Cutting-Edge performance** Built on JAX, leveraging Just-In-Time (JIT) for high-performance array-based operations on the CPU, GPU, and TPU.
* **Solve inverse problems with ease:** All internals are automatically differentiable, enabling model diagnosis and gradient-based optimization (e.g., reducing the need to solve the tangential stiffness tensor).
* **Modular & Extendable:** The modular structure, coupled with batched and parallelized continuum mechanics operations, will enable modification and extension of existing components, while also enabling rapid prototyping.

## 🚀 Features

#### MPM Solver

* **Formulation:** Explicit dynamic solver utilizing the Update Stress Last (USL) integration scheme.
* **Schemes:** Supports multiple transfer methods, including FLIP/PIC (Fluid Implicit Particle/Particle-In-Cell hybrid), Affine Particle-In-Cell (APIC), and Affine FLIP (AFLIP).
* **Shape functions**: Supports various basis functions: Linear, Quadratic, and Cubic B-splines.
* **Rigid bodies**: Includes a penalty-based algorithm for modeling frictional contact with rigid boundaries or objects.
*   **Adaptive Time Stepping:** Employs adaptive time stepping based on the Courant–Friedrichs–Lewy (CFL) condition to ensure numerical stability.
*   **Particle-Grid Mapping:** Utility functions seamlessly transfer data between particle and grid representations.
*   **Dimensionality** Capability to solve both 3D and plane strain problems.
* **Boundary conditions:** slip and no slip boundaries


#### Constitutive models
* **Built-in models:**
  * Non-associated Drucker Prager (with optional cohesion).
  * Modified Cam-Clay (double-logarithmic ln v - ln p  formulation). 
  * Newtonian Fluid
  * $\mu (I)$ rheology (incorporating linear pressure-dependent compression).
* **Upcoming models:** Unified Hardning, and Critical State Unified Hardening models
* **Available SIP Tests:**  
    *   Triaxial Compression/Extension (Drained/Undrained)
    *   Constant Pressure Shear
    *   Constant Volume Shear
    *   Isotropic Compression

## 💻 Installation Instructions
- Install uv [here](https://docs.astral.sh/uv/getting-started/installation/)
- Clone repository `git clone git@github.com:GrainLearning/HydraxMPM.git && cd HydraxMPM`
- Install dependencies `uv sync`
- Run an example, e.g., `uv run examples/dambreak/dambreak.py`. Output is found in the `./examples/dambreak/` directory.

## 👥 Contributors:

* Retief Lubbe (Soil Micro Mechanics group / University of Twente)
* Hongyang Cheng (Soil Micro Mechanics group / University of Twente)

## 🙏 cknowledgements
This research is part of the project TUSAIL [Training in Upscaling Particle Systems: Advancing Industry across Length-scales](https://tusail.eu)  and has received funding from the European Horizon2020 Framework Programme for research, technological development and demonstration under grant agreement ID 955661.

> [!WARNING]  
> This is a research software under active development (pre-alpha). APIs and functionality are subject to change without notice.

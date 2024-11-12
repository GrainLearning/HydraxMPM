
A Material Point Method solver developed with [JAX](https://jax.readthedocs.io/en/latest/quickstart.html)

This code is still in pre-alpha, i.e., all models may not be feature-complete.


## Current features
Basic shape functions (linear, cubic)
Solvers (USL, APIC)
Materials ( Drucker Prager, Linear isotropic Elastic, Modified Cam Clay, Newtonian Fluid, $\mu (I)$ rheology)
Forces (Rigid body contact, gravity, slip and no slip boundaries)

## Installation instructions
- Install uv [here](https://docs.astral.sh/uv/getting-started/installation/)
- Clone repository `git clone git@github.com:GrainLearning/HydraxMPM.git && cd HydraxMPM`
- For GPU `uv sync --group plot`
- For CPU `uv sync --group cpu plot`
- Run an example, e.g., `uv run examples/dambreak/dambreak.py`. Output is found in the `./examples/dambreak/` directory.


## Contributors:
Retief Lubbe, Hongyang Cheng

University of Twente, SMM

## Acknowledgements
This research is part of the project TUSAIL [Training in Upscaling Particle Systems: Advancing Industry across Length-scales](https://tusail.eu)  and has received funding from the European Horizon2020 Framework Programme for research, technological development and demonstration under grant agreement ID 955661.


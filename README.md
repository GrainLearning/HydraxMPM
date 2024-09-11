
Differentiable Material Point Method solver developed with [JAX](https://jax.readthedocs.io/en/latest/quickstart.html)

This is a pre-release version that is still in development.


## Current features
Basic shape functions (linear, cubic)
Solvers (USL, APIC)
Materials ( Drucker Prager, Linear isotropic Elastic, Modified Cam Clay, Newtonian Fluid, $\mu (I)$ rheology)
Forces (Rigid body contact, gravity, slip and no slip boundaries)

## Installation instructions
- Clone repository
- Install poetry [here](https://python-poetry.org/docs/)
- `poetry install`
- `poetry shell`

Examples that should work
- Cube fall 3D, Cube fall, dambreak, sphere impact, CHOPS2024
- To run example, run `python <name of script>`

## Authors:
Retief Lubbe, Hongyang Cheng, Stefan Luding, Vanessa Magnanimo

University of Twente, SMM & M&S

## Acknowledgements


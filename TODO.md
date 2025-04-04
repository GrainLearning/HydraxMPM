
## Core



## Documentation


### How to guides

- [ ] How to install

- [ ] How to create a custom constitutive model

    -[ ] Example MU I / Phi I

- [ ] How to create a custom element benchmark

- [ ] How to visualize results

- [ ] How to calibrate model...

### Tutorials
- [ ] Granular column collapse

### References

- [ ] Examples
- [ ] Unit tests
- [ ] 3D 2D


### Explaination
- [ ] Motivation
- [ ] JAX
- [ ] Philsophy
- [ ] Model backgroud

- [ ] Remove support for init by density..
    - [ ] Add documentation note, it can be done to overwrite init_state method...

### Supported laws
- [ ] Drucker-Prager
- [ ] Modified Cam Clay
- [ ] Mu I rheology


## Unoffical
- [ ] UH model**

- Fix post update for energies
- ADD FAQ on how to use CPU GPU os.env flat



- Add error handler in mpm solver to output data if throws error
    - After cleanup
- write some standard checks, unit tests, to see if models are functioning properly
    - TRX compression
    - ISO extension
- Make a list of experimental validation simulations
- Make file to run all checks

- Open questions
    - How to organize plots? worth automated way


- Add note that large velocity gradients may be due to, a large spatial variation in the materila points velocity..
    - particles moving opposite direction, or shearing..
    Excessively large gradients can destabilize simulations (e.g., causing unphysical stresses or grid artifacts)




- [ ] Rename ETSolver, ETBenchmarks to SIP
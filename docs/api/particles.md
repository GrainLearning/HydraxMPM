<!-- 
# Particles
!!! abstract "" 
    This module gives access to the material point method lagrangian markers (called particles).

!!! example

    A particles class is created by specifying an MPM config and a array of positions:


    ```py
        import hydraxmpm as hdx
        
        config = hdx.MPMConfig(...)

        position_stack = jnp.array([...])

        hdx.Particles(config=config,position_stack)

    ```

    Utility functions are designed to work well with the particles class to enable easy post-processing

    ```py
    ...
    stress_stack = hdx.get_pressure_stack(particles.stress_stack)
    ```

::: particles.particles -->

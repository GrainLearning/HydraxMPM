<!-- TODO see literature on shapefunctions to add more details

# Shape functions
!!! abstract ""
    Shapefunctions interpolate values from the material points to the background grid and back.
    The shapefunction quantites are stored in this class. Two shapefunctions are currently implemented, namely, cubic splines and linear shape functions [^1].

!!! note
    It is important to specify the `shapefuction_type` in `MPMConfig` before creating the shapefunction class.


::: shapefunctions.shapefunctions

# Cubic splines
!!! abstract ""
    It is recommended that each background cell is populated by
    2 (1D), 4 (2D), 8 (3D) material points. The optimal integration points are at $x_1=0.2113$ and $x_2=0.7887$ determined by Gauss quadrature rule. The cubic spline shapefunction communicates with 16 (2D) or 64 (3D) surrounding nodes per particles, which makes it costly to run.

!!! example

    ```py
        import hydraxmpm as hdx

        config = hdx.MPMConfig(
            shapefunction_type="cubic"
        )

        hdx.CubicShapeFunction(config=config)

    ```
<!-- !!! reference function to generate material point -->

<!-- ::: shapefunctions.cubic -->


# Linear

!!! abstract ""
    This is the simplest which is discontinious and prone to instabilities.


!!! example

    ```py
        import hydraxmpm as hdx

        config = hdx.MPMConfig(
            shapefunction="linear"
        )

    ```

<!-- Add reference -->
<!-- ::: shapefunctions.linear -->


------------
**References:**

[^1]: De Vaucorbeil, Alban, et al. 'Material point method after 25 years: theory, implementation, and applications.' -->

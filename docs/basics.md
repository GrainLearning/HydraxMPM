# The Basics



# Code structure

HydraxMPM follows an array of structures design. That is, a set of scalars has a shape `(number of material points,)`, vectors `(number of material points, dimension)`, tensors `(number of material points, dimension)`. These arrays are typically denoted as stacks (e.g., `mass_stack`, `velocity_stack`). Some arrays are always defined in 3D, e.g., the deformation gradient `F_stack`, which has a shape `(number of material points,3)`.


- Objects are defined as [dataclasses](https://docs.python.org/3/library/dataclasses.html), common practice in JAX

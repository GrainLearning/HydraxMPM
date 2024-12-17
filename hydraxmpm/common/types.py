from jaxtyping import Array, Float, Int, UInt

from typing_extensions import Self, Union


TypeFloat = Float[Array, "..."]
TypeFloatVector = Float[Array, "dim"]
TypeFloatMatrix3x3 = Float[Array, "3 3"]

# Particles
TypeFloatScalarPStack = Float[Array, "num_points"]
TypeFloatVectorPStack = Float[Array, "num_points dim"]
TypeFloatMatrixPStack = Float[Array, "num_points dim dim"]
TypeFloatMatrix3x3PStack = Float[Array, "num_points 3 3"]

# Nodes
TypeFloatScalarNStack = Float[Array, "num_nodes"]
TypeFloatVectorNStack = Float[Array, "num_nodes dim"]
TypeFloatMatrixNStack = Float[Array, "num_nodes dim dim"]
TypeFloatMatrix3x3NStack = Float[Array, "num_nodes 3 3"]

# Any
TypeFloatScalarAStack = Float[Array, "*"]
TypeFloatVectorAStack = Float[Array, "* dim"]
TypeFloatVector3AStack = Float[Array, "* 3"]
TypeFloatMatrixAStack = Float[Array, "* dim dim"]
TypeFloatMatrix3x3AStack = Float[Array, "* 3 3"]
TypeIntScalarAStack = Int[Array, "*"]
TypeUIntScalarAStack = UInt[Array, "*"]

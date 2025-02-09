from typing import Optional, Self, Tuple, Union

from ..common.base import Base
from ..common.types import (
    TypeFloat,
    TypeFloatMatrixPStack,
    TypeFloatScalarAStack,
    TypeInt,
)
from ..material_points.material_points import MaterialPoints


class ConstitutiveLaw(Base):
    rho_p: Optional[Union[TypeFloatScalarAStack, TypeFloat]] = 1.0
    rho_0: Optional[Union[TypeFloatScalarAStack, TypeFloat]] = None
    p_0: Optional[Union[TypeFloatScalarAStack, TypeFloat]] = None
    d: Optional[Union[TypeFloatScalarAStack, TypeFloat]] = None

    eps_e_stack: Optional[TypeFloatMatrixPStack] = None

    def __init__(self, **kwargs):
        self.d = kwargs.get("d", None)
        self.p_0 = kwargs.get("p_0", None)
        phi_0 = kwargs.get("phi_0", None)
        self.rho_p = kwargs.get("rho_p", 1.0)

        if phi_0 is None:
            self.rho_0 = kwargs.get("rho_0", 1.0)
        else:
            self.rho_0 = self.rho_p * phi_0
        super().__init__(**kwargs)

    def update(
        self: Self,
        material_points: MaterialPoints,
        dt: TypeFloat,
        dim: Optional[TypeInt] = 3,
    ) -> Tuple[MaterialPoints, Self]:
        pass

    def init_state(
        self: Self,
        material_points: MaterialPoints,
        **kwargs,
    ) -> Tuple[Self, MaterialPoints]:
        p_0 = self.p_0
        if p_0 is None:
            p_0 = material_points.p_stack

        material_points = material_points.init_stress_from_p_0(p_0)

        material_points = material_points.init_mass_from_rho_0(self.rho_0)
        params = self.__dict__

        params.update(
            p_0=p_0,
            **kwargs,
        )
        return self.__class__(**params), material_points

    @property
    def phi_0(self):
        """Assumes dry case"""
        return self.rho_0 / self.rho_p

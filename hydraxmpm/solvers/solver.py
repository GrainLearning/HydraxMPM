import equinox as eqx

from ..config.mpm_config import MPMConfig


class Solver(eqx.Module):
    
    config: MPMConfig = eqx.field(static=True)
    
    def __init__(self,config: MPMConfig):
        self.config = config
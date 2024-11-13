
import equinox as eqx

from ..config.mpm_config import MPMConfig


class Material(eqx.Module):
    """Base material class."""
    config: MPMConfig = eqx.field(static=True)
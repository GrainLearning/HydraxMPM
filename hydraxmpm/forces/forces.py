import equinox as eqx

from ..config.mpm_config import MPMConfig


class Forces(eqx.Module):
    """Force state for the material properties."""

    config: MPMConfig = eqx.field(static=True)

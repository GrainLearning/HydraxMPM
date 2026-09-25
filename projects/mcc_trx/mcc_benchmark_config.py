"""Shared settings for the Modified Cam Clay triaxial benchmark."""

from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "benchmark_data"

# Loading and initial state (SI units).
CONFINE = 10_000.0  # Pa
AXIAL_RATE = 0.02  # s^-1, compression positive
NUM_STEPS = 10_000
DT = 0.01  # s
OCR = 1.0
SENSITIVITY_AXIAL_STRAINS = (0.01, 0.05, 0.10, 0.20)
SENSITIVITY_PROBE_SCALE = 1.0e-3

# Modified Cam Clay parameters.
NU = 0.3
M = 0.9
LAM = 0.2
KAP = 0.05
N = 2.0
P_REF = 1_000.0  # Pa
RHO_P = 2_650.0  # kg/m^3


def axial_strain_samples():
    """Return the initial state followed by every applied strain increment."""
    import numpy as np

    increment = AXIAL_RATE * DT
    return np.arange(NUM_STEPS + 1, dtype=float) * increment

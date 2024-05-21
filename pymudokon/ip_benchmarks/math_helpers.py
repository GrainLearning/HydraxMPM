import numpy as np


def get_pressure(stress_list):
    pressure_list = -(1 / 3.0) * (
        stress_list[:, 0, 0] + stress_list[:, 1, 1] + stress_list[:, 2, 2]
    )
    return pressure_list


def get_dev_stress(stress_list, pressure_list=None):
    if pressure_list is None:
        pressure_list = get_pressure(stress_list)

    dev_stress_list = stress_list + np.identity(3) * pressure_list[:, None, None]
    return dev_stress_list


def get_q_vm(stress_list, dev_stress_list=None):
    if dev_stress_list is None:
        dev_stress_list = get_dev_stress(stress_list)

    q_vm_list = np.array(
        list(
            map(
                lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
                dev_stress_list,
            ),
        ),
    )
    return q_vm_list


def get_tau(stress_list, dev_stress_list=None):
    if dev_stress_list is None:
        dev_stress_list = get_dev_stress(stress_list)

    tau_list = np.array(
        list(
            map(
                lambda s: 0.5 * np.trace(s @ s.T),
                dev_stress_list,
            ),
        ),
    )
    return tau_list


def get_volumetric_strain(strain_list):
    volumetric_strain_list = -(
        strain_list[:, 0, 0] + strain_list[:, 1, 1] + strain_list[:, 2, 2]
    )
    return volumetric_strain_list


def get_dev_strain(strain_list, volumetric_strain_list=None):
    if volumetric_strain_list is None:
        volumetric_strain_list = get_volumetric_strain(strain_list)

    dev_strain_list = (
        strain_list + (1.0 / 3) * np.identity(3) * volumetric_strain_list[:, None, None]
    )
    return dev_strain_list


def get_gamma(strain_list, dev_strain_list=None):
    if dev_strain_list is None:
        dev_strain_list = get_dev_strain(strain_list)

    gamma_list = np.array(
        list(
            map(
                lambda s: 0.5 * np.trace(s @ s.T),
                dev_strain_list,
            ),
        ),
    )
    return gamma_list

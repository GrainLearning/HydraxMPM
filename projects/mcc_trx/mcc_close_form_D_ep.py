# encoding: utf-8
import sys

import numpy as np

# Cam-clay critical state parameters
pc_0 = 10.0
M = 0.9
lambda_val = 0.2
kappa = 0.05
N = 2.0
nu = 0.3
p_ref = 1.0

# Initial conditions
p0 = 10.0
V = N * (p_ref / pc_0) ** lambda_val * (pc_0 / p0) ** kappa
void_ratio_0 = V - 1
Phi_0 = 1.0 / (1.0 + void_ratio_0)
deformation_mode = "undrained" if "--undrained" in sys.argv else "drained"

# Loading history as mcc.py
eqp_inc = 0.02 * 0.01
load_length = 10_000 + 1
dt = 0.01
eqp_inc_history = np.full(load_length, eqp_inc)

# Dynamic contribution is disabled for this benchmark.
Delta_Phi = 0.0
M_c = 0.0

# Declarations
void_ratio_total = np.zeros(load_length)
void_ratio_q = np.zeros(load_length)
void_ratio_cp = np.zeros(load_length)
p = np.zeros(load_length)
q = np.zeros(load_length)
u = np.zeros(load_length)
ev = np.zeros(load_length)
ev_cp = np.zeros(load_length)
ev_qp = np.zeros(load_length)
eq = np.zeros(load_length)
d_p_c = np.zeros(load_length)
p_c = np.zeros(load_length)
q_c = np.zeros(load_length)
p_total = np.zeros(load_length)
q_total = np.zeros(load_length)
pc_history = np.zeros(load_length)
tangent_history = np.zeros((load_length, 2, 2))

# Initialize state variables
pc = pc_0
K_c = 0.0
G_c = 0.0
p[0] = p0
p_total[0] = p0
pc_history[0] = pc_0
sigma_q = np.array([p0, p0, p0, 0.0, 0.0, 0.0])
sigma_c = np.zeros(6)
sigma = sigma_q + sigma_c
epsilon = np.zeros(6)
void_ratio_total[0] = void_ratio_0
void_ratio_q[0] = void_ratio_0
void_ratio_cp[0] = void_ratio_0
yield_surf = (q[0] ** 2 / M**2 + p[0] ** 2) - p[0] * pc

# Load-step cycle and analytical stiffness update
for i, current_eqp_inc in enumerate(eqp_inc_history[:-1]):
    d_eqp_inc = current_eqp_inc - eqp_inc_history[i - 1] if i > 0 else 0.0
    De = np.zeros((6, 6))
    De_c = np.zeros((6, 6))
    D_c = np.zeros((6, 6))
    df_ds = np.zeros((6, 1))
    df_dep = np.zeros((6, 1))

    # Bilogarithmic elastic moduli
    K = p[i] / kappa
    G = 3 * K * (1 - 2 * nu) / (2 * (1 + nu))

    if yield_surf == 0:
        pc = (q[i] ** 2 / M**2 + p[i] ** 2) / p[i]
    else:
        pc = pc_0
    pc_history[i + 1] = pc

    for m in range(6):
        for n in range(6):
            if m <= 2:
                if yield_surf < 0:
                    df_ds[m, 0] = 0
                    df_dep[m, 0] = 0
                else:
                    df_ds[m, 0] = (
                        (2 * p[i] - pc) / 3
                        + 3 * (sigma_q[m] - p[i]) / M**2
                    )
                    df_dep[m, 0] = -p[i] * pc / (lambda_val - kappa)
                if m == n:
                    De[m, n] = K + 4 / 3 * G
                    De_c[m, n] = K_c
                    D_c[m, n] = 4 / 3 * G_c
                elif n <= 2:
                    De[m, n] = K - 2 / 3 * G
                    De_c[m, n] = K_c
                    D_c[m, n] = -2 / 3 * G_c
            if m > 2:
                df_ds[m, 0] = 0
                df_dep[m, 0] = 0
                if m == n:
                    De[m, n] = G
                    D_c[m, n] = G_c

        if yield_surf < 0:
            D_q = De
        else:
            D_q = De - (
                De.dot(df_ds).dot(df_ds.T).dot(De)
                / (-(df_dep.T).dot(df_ds) + (df_ds.T).dot(De).dot(df_ds))
            )

    # d(p,q)/d(eps_a,eps_r), converted from kPa to Pa
    strain_map = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    invariant_map = np.array(
        [[1 / 3, 1 / 3, 1 / 3], [1.0, -0.5, -0.5]]
    )
    tangent_history[i] = invariant_map @ D_q[:3, :3] @ strain_map * 1_000

    if deformation_mode == "drained":
        ratio = -D_q[1, 0] / (D_q[1, 1] + D_q[1, 2])
        d_epsilon = np.array(
            [
                current_eqp_inc,
                ratio * current_eqp_inc,
                ratio * current_eqp_inc,
                0,
                0,
                0,
            ]
        )
        d_d_epsilon = np.array(
            [d_eqp_inc, ratio * d_eqp_inc, ratio * d_eqp_inc, 0, 0, 0]
        )
    else:
        d_epsilon = np.array(
            [current_eqp_inc, -current_eqp_inc / 2, -current_eqp_inc / 2, 0, 0, 0]
        )
        d_d_epsilon = np.array(
            [d_eqp_inc, -d_eqp_inc / 2, -d_eqp_inc / 2, 0, 0, 0]
        )

    de_v = np.sum(d_epsilon[:3])
    d_eqp_dev = d_d_epsilon - np.r_[
        np.full(3, np.sum(d_d_epsilon[:3]) / 3), np.zeros(3)
    ]
    d_eqp_inc_mag = np.sqrt(
        2 / 3
        * (
            d_eqp_dev[:3].dot(d_eqp_dev[:3])
            + 2 * d_eqp_dev[3:].dot(d_eqp_dev[3:])
        )
    )
    d_eqp_inc_eff = np.sign(d_eqp_inc) * d_eqp_inc_mag

    # Dynamic contribution to the stress increment
    if d_eqp_inc_mag != 0 and Delta_Phi != 0:
        Phi = 1.0 / (1.0 + void_ratio_total[i])
        Phi_c = 1.0 / (1.0 + void_ratio_cp[i])
        de_v_c = -Phi / Phi_c**2 * Delta_Phi * d_eqp_inc_eff / dt
        d_p_c[i + 1] = (
            p[i] / (lambda_val * Phi**2) * Delta_Phi * d_eqp_inc_eff / dt
        )
        K_c = d_p_c[i + 1] / de_v_c
        G_c = d_p_c[i + 1] * M_c / d_eqp_inc_eff / 3
    else:
        de_v_c = 0.0
        K_c = 0.0
        G_c = 0.0

    d_epsilon_v_c = np.r_[np.full(3, de_v_c / 3), np.zeros(3)]
    if yield_surf < 0:
        de_v_p = 0.0
    else:
        numerator = df_ds.T.dot(De).dot(d_epsilon - d_epsilon_v_c)
        denominator = -df_dep.T.dot(df_ds) + df_ds.T.dot(De).dot(df_ds)
        d_epsilon_p = float(numerator[0] / denominator[0, 0]) * df_ds
        de_v_p = float(np.sum(d_epsilon_p[:3]))

    # Update stress and strain
    d_sigma_q = D_q.dot(d_epsilon - d_epsilon_v_c)
    d_sigma_c = De_c.dot(d_epsilon_v_c) + D_c.dot(d_eqp_dev)
    sigma_q += d_sigma_q
    sigma_c += d_sigma_c
    sigma = sigma_q + sigma_c
    epsilon += d_epsilon
    p_total[i + 1] = np.sum(sigma[:3]) / 3
    p_s = sigma - np.array([1.0, 1.0, 1.0, 0, 0, 0]) * p_total[i + 1]
    q_total[i + 1] = np.sqrt(3 / 2 * p_s.dot(p_s))

    p_c[i + 1] = np.sum(sigma_c[:3]) / 3
    p_s_c = sigma_c - np.array([1.0, 1.0, 1.0, 0, 0, 0]) * p_c[i + 1]
    q_c[i + 1] = np.sqrt(3 / 2 * p_s_c.dot(p_s_c))

    ev[i + 1] = np.sum(epsilon[:3])
    epsilon_s = (
        epsilon - np.array([1.0, 1.0, 1.0, 0, 0, 0]) * ev[i + 1] / 3
    )
    eq[i + 1] = np.sqrt(2 / 3 * epsilon_s.dot(epsilon_s))
    ev_cp[i + 1] = ev_cp[i] + de_v_c
    ev_qp[i + 1] = ev_qp[i] + de_v_p

    p[i + 1] = np.sum(sigma_q[:3]) / 3
    p_q_s = sigma_q - np.array([1.0, 1.0, 1.0, 0, 0, 0]) * p[i + 1]
    q[i + 1] = np.sqrt(3 / 2 * p_q_s.dot(p_q_s))
    u[i + 1] = p0 + q_total[i + 1] / 3 - p_total[i + 1]

    V = N * (p_ref / pc) ** lambda_val * (pc / p[i + 1]) ** kappa
    void_ratio_q[i + 1] = V - 1
    void_ratio_cp[i + 1] = void_ratio_cp[i] - (
        1 + void_ratio_total[i]
    ) * de_v_c
    void_ratio_total[i + 1] = void_ratio_total[i] - (
        1 + void_ratio_total[i]
    ) * de_v

    if yield_surf < 0:
        yield_surf = (
            q[i + 1] ** 2 + M**2 * p[i + 1] ** 2 - M**2 * p[i + 1] * pc
        )
    else:
        yield_surf = 0

tangent_history[-1] = tangent_history[-2]
axial_strain = np.arange(load_length) * eqp_inc
radial_strain = 0.5 * (ev - axial_strain)
pc_surface = (q**2 / M**2 + p**2) / p

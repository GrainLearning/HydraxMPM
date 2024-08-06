import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

colors = ["red", "blue", "green", "purple", "orange", "black"]


import os
import scienceplots

plt.style.use(["science", "no-latex"])
# Plot params
os.chdir(os.path.dirname(__file__))


# d = 1, rhop = 1, e = 0.7, mu = 0, K = 10^3, shear rate = 1
names = ["nu", "p", "s", "T", "I", "CN", "k"]
df_k3 = pd.read_table("./DATA_SS/k1ex.data", sep=" ", skiprows=2, nrows=21, names=names)
df_k4 = pd.read_table("./DATA_SS/k1ex.data", sep=" ", skiprows=26, nrows=21, names=names)
df_k5 = pd.read_table("./DATA_SS/k1ex.data", sep=" ", skiprows=50, nrows=21, names=names)
df_k6 = pd.read_table("./DATA_SS/k1ex.data", sep=" ", skiprows=74, nrows=26, names=names)
df_k7 = pd.read_table("./DATA_SS/k1ex.data", sep=" ", skiprows=103, nrows=21, names=names)
df_k8 = pd.read_table("./DATA_SS/k1ex.data", sep=" ", skiprows=127, nrows=21, names=names)


def df_format(df, d, K):
    df["pstar"] = df["p"] * (d / K)
    df["sstar"] = df["s"] * (d / K)
    df["qstar"] = df["sstar"] * np.sqrt(3)
    df["mu"] = df["s"] / df["p"]
    df["e"] = 1.0 / df["nu"] - 1
    df["stiffness"] = K
    return df


df_k3 = df_format(df_k3, d=1, K=10**3)
df_k4 = df_format(df_k4, d=1, K=10**4)
df_k5 = df_format(df_k5, d=1, K=10**5)
df_k6 = df_format(df_k6, d=1, K=10**6)
df_k7 = df_format(df_k7, d=1, K=10**7)
df_k8 = df_format(df_k8, d=1, K=10**8)

dem_stiffness_all = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

df_dem_all = [
    df_k3,
    df_k8,
]


# Model Parameters
phi_c = 0.634
mu_s = 0.12
mu_d = 0.55
I0 = 0.2
d = 1.0
p_phi = 0.33
rho_p = 1.0
I_phi = 3.28
dgamma_dt = 1.0
p0 = 0.9

lam = 0.04


def get_PI0(phi):
    if phi > phi_c:
        return 0
    top = dgamma_dt**2 * d**2 * rho_p
    bot = I_phi**2 * np.log(phi_c / phi) ** 2
    return top / bot


def get_PQS0(phi):
    if phi < phi_c:
        return 0
    return (phi / phi_c) * np.exp(-lam)


def get_PI(phi, pqs, k):
    PI0 = get_PI0(phi)
    PQS0 = get_PQS0(phi)

    top = dgamma_dt**2 * d**2 * rho_p
    bot = I_phi**2 * np.log(phi_c / phi) ** 2 - lam * np.log(pqs + PI0)

    return top / bot - PQS0


def get_PQS(phi):
    if phi < phi_c:
        return 0
    return (phi / phi_c) * np.exp(-lam)


# def get_I(p):
#     return (dgamma_dt * d) / np.sqrt(p / rho_p)


# def get_mu_I(I):
#     return mu_s + (mu_d - mu_s) / (1 + I0 / I)


# def get_mu_I_correction(mu_I, p, k):
#     p_star = p * (d / k)
#     return mu_I * (1.0 - (p_star / p0) ** 0.5)


# def add_pressures(p_qs, p_inert):
#     # return (p_qs * p_inert) / (p_qs**2 + +(p_inert**2))
#     return p_qs + p_inert


# def get_p_implicit(p_inert, k, p_qs, phi):
#     # p_total = p_inert + p_qs
#     p_total = add_pressures(p_qs, p_inert)
#     I = get_I(p_total)

#     # term1 = phi_c * (1 - I / I_phi) * (1.0 + p_star / p_phi)
#     # term2 = phi
#     sol = lam * np.log(p_total) + np.log(phi_c / phi) - I / I_phi

#     return sol


def solve_p_I(phi_stack, p_qs_stack, k):
    p_stack = []
    I_stack = []
    mu_stack = []
    q_stack = []

    for i, phi in enumerate(phi_stack):
        sol = scipy.optimize.fsolve(get_p_implicit, 0.001, args=(k, p_qs_stack[i], phi))
        p = sol
        # print(sol)
        I = get_I(p)

        mu_I = get_mu_I(I)

        mu_I_corr = get_mu_I_correction(mu_I, p, k)

        q = mu_I_corr * p * np.sqrt(3)

        p_stack.append(p)
        I_stack.append(I)

        q_stack.append(q)
        mu_stack.append(mu_I_corr)

    I_stack = np.array(I_stack)
    p_stack = np.array(p_stack)
    mu_stack = np.array(mu_stack)
    q_stack = np.array(q_stack)

    return p_stack, I_stack, mu_stack, q_stack


def get_p_qs(phi):
    if phi < phi_c:
        return 0
    return (phi / phi_c) * np.exp(-lam)


for i, df_raw in enumerate(df_dem_all):
    k = dem_stiffness_all[i]

    # df = df_raw.loc[df_raw["nu"] >= phi_lower]

    nu_stack = df_raw["nu"].values

    df_raw["p_sim_qs"] = [get_p_qs(nu) for nu in nu_stack]
    df_raw["p_sim_qs_star"] = df_raw["p_sim_qs"] / k

    p_stack, I_stack, mu_stack, q_stack = solve_p_I(
        nu_stack, df_raw["p_sim_qs"].values, k
    )
    df_raw["p_sim"] = p_stack

    # df_raw["I_sim"] = I_stack
    # df_raw["q_sim"] = q_stack
    # df_raw["mu_sim"] = mu_stack

    df_raw["p_sim_star"] = df_raw["p_sim"] / k

    df_raw["p_sim_total_star"] = df_raw["p_sim_qs_star"] + df_raw["p_sim_star"]
    # df_raw["q_sim_star"] = q_stack / k

    print(df_raw["p_sim_qs"].min())
# # plotting
# fig, ax = plt.subplots(1, 2, sharey=True, figsize=(11, 5))
# fig.subplots_adjust(wspace=0.0)
# fig_nu, ax_nu = plt.subplots(1, 2, figsize=(10, 5))
# # labels = []
fig_p_nu, ax_p_nu = plt.subplots(figsize=(10, 10))
# fig_mu_nu, ax_mu_nu = plt.subplots()
for i, df_raw in enumerate(df_dem_all):
    k = dem_stiffness_all[i]
    label = "K={:.2e}".format(k)

    df = df_raw.loc[df_raw["nu"] >= 0.5]

    # df_above = df.loc[df["nu"] >= phi_c]
    # df_bellow = df.loc[df["nu"] < phi_c]
    df.plot(
        x="nu",
        y="pstar",
        ax=ax_p_nu,
        c=colors[i],
        label=f"K = {k:.2e}",
        kind="scatter",
    )
    df.plot(
        x="nu",
        y="p_sim_qs_star",
        ax=ax_p_nu,
        c=colors[i],
        label=f"K = {k:.2e}",
        marker="x",
    )

    df.plot(
        x="nu",
        y="p_sim_total_star",
        ax=ax_p_nu,
        c=colors[i],
        label=f"K = {k:.2e}",
        kind="scatter",
        marker="^",
    )
ax_p_nu.get_legend().remove()
ax_p_nu.set_xlim(0.5, None)
ax_p_nu.set_yscale("log")
ax_p_nu.set_xlim(0.5, None)
plt.show()
#     # experimental data
#     df_bellow.plot(
#         x="pstar", y="qstar", ax=ax[0], kind="scatter", label=None, c=colors[i]
#     )
#     df_above.plot(
#         x="pstar", y="qstar", ax=ax[1], c=colors[i], label=f"K = {k:.2E}", kind="scatter"
#     )
#     # simulation data
#     df_bellow.plot(
#         x="p_sim_star",
#         y="q_sim_star",
#         ax=ax[0],
#         # kind="scatter",
#         ls="--",
#         c=colors[i],
#     )
#     df_above.plot(
#         x="p_sim_star",
#         y="q_sim_star",
#         ax=ax[1],
#         c=colors[i],
#         # label=f"K = {k:.2E}",
#         label=None,
#         # kind="scatter",
#         ls="--",
#     )
#     # experiment
#     df.plot(
#         x="nu",
#         y="pstar",
#         ax=ax_p_nu,
#         c=colors[i],
#         label=f"K = {k:.2e}",
#         kind="scatter",
#     )
#     df.plot(
#         x="nu", y="mu", ax=ax_mu_nu, c=colors[i], label=f"K = {k:.2e}", kind="scatter"
#     )
#     # simulation
#     df.plot(
#         x="nu",
#         y="p_sim_star",
#         ax=ax_p_nu,
#         c=colors[i],
#         # label=f"K = {k:.2e}",
#         ls="--",
#     )
#     df.plot(x="nu", y="mu_sim", ax=ax_mu_nu, c=colors[i], ls="--")

# ax[0].set_title("$\phi < \phi_J$")

# ax[0].set_xlim(0, None)
# ax[0].invert_xaxis()
# ax[0].set_ylabel(r"$q^*$")
# ax[0].set_xlabel(r"$p^*$ (inverted)")


# ax[1].set_title("$\phi > \phi_J$")
# ax[1].set_xlim(0, None)
# ax[1].set_xlabel(r"$p^*$")


# ax_p_nu.set_xlim(0.5, None)
# ax_p_nu.set_yscale("log")
# ax_p_nu.set_xlim(0.5, None)

# ax[0].get_legend().remove()
# ax[1].get_legend().remove()

# ax_p_nu.get_legend().remove()
# ax_p_nu.set_ylabel(r"$p^*$")
# ax_p_nu.set_xlabel(r"$\phi$")

# ax_mu_nu.get_legend().remove()
# ax_mu_nu.set_ylabel(r"$\mu$")
# ax_mu_nu.set_xlabel(r"$\phi$")
# # lgd = ax[1].legend(labels=labels, ncols=3, loc="upper left", bbox_to_anchor=(-0.5, 0.5))


# # plt.show()
# fig.savefig("p-q-plot.png")
# # fig_nu.savefig("nu-p-plot.png")

# ax_mu_nu.figure.savefig("p_nu.png")
# ax_p_nu.figure.savefig("mu_nu.png")

# # # # plot inverted p-q graph

# ////////////////

# for i, df_raw in enumerate(df_dem_all):
#

#     # p = sol
#
#     I_vals = []
#     p_vals = []
#     mu_vals = []
#     q_vals = []
#     for nu in nu_vals:
#         p, I = solve_p(nu, K)
#         p_vals.append(p)
#         I_vals.append(I)

#         mu_I = get_mu_I(I)
#         mu_I_corr = get_mu_I_correction(mu_I, K)
#         mu_vals.append(mu_I_corr)

#         q = mu_I_corr * p / np.sqrt(3)
#         q_vals.append(q)

#     I_vals = np.array(I_vals)
#     p_vals = np.array(p_vals)
#     mu_vals = np.array(mu_vals)
#     q_vals = np.array(q_vals)

#     id_above = np.where(nu_vals >= phi_c)[0]
#     id_bellow = np.where(nu_vals < phi_c)[0]


#     df.plot(
#         x="nu", y="pstar", ax=ax_nu[0], c=colors[i], label=f"K = {K:.2e}", kind="scatter"
#     )
#     df.plot(
#         x="nu", y="mu", ax=ax_nu[1], c=colors[i], label=f"K = {K:.2e}", kind="scatter"
#     )

#     # print(p_vals)
#     df = df.loc[df_raw["nu"] >= phi_lower]
#     df_above = df.loc[df["nu"] >= phi_c]
#     df_bellow = df.loc[df["nu"] < phi_c]
#     df_bellow.plot(x="pstar", y="qstar", ax=ax[0], kind="scatter", c=colors[i])

#     df_above.plot(
#         x="pstar", y="qstar", ax=ax[1], c=colors[i], label=f"K = {K:.2E}", kind="scatter"
#     )


# plt.show()
# # def fn(p, args):
# #     phi = args
# #     phi_c = 0.634
# #     p_phi = 0.33
# #     I_phi = 3.28
# #     rho_p = 1
# #     dgamma_dt = 1
# #     d = 1
# #     k = 10**3

# #     lam = 0.1
# #     term1 = lam * jnp.log(p) + I_phi * jnp.log(phi_c / phi)
# #     # term1 = p*(d/k)/p_phi + I_phi*jnp.log(phi_c / phi)
# #     term2 = (dgamma_dt * d) / jnp.sqrt(p / rho_p)
# #     sol = term1 - term2
# #     return sol


# # solver = optx.Newton(rtol=1e-3, atol=1e-8)

# # # y0 = jnp.array([.3])

# # fig, ax = plt.subplots()
# # phi_stack = jnp.arange(0.3, 0.7, 0.01)
# # p_stack = []
# # for phi in phi_stack:
# #     sol = optx.root_find(fn, solver, 0.3, args=phi, throw=False)

# #     p_stack.append(sol.value)


# # phi_stack = np.array(phi_stack)
# # # ax.plot(phi_stack, p_stack)
# # ax.plot(phi_stack, np.nan_to_num(p_stack) / 10**3, ls="--", lw=2, label="Impl")
# # df_k8.plot(y="pstar", x="nu", kind="scatter", ax=ax, label="K = 10^8")
# # df_k3.plot(y="pstar", x="nu", kind="scatter", ax=ax, label="K = 10^3", color="red")
# # ax.set_yscale("log")
# # # ax.set_xlim(0.5,0.7)
# # # ax.set_ylim(0,10e7)

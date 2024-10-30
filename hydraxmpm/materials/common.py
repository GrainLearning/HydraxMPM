import jax.numpy as jnp


def get_bulk_modulus(E, nu):
    return E / (3.0 * (1.0 - 2.0 * nu))


def get_shear_modulus(E, nu):
    return E / (2.0 * (1.0 + nu))


def get_lame_modulus(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def get_lin_elas_dev(eps_e_dev, G):
    return 2.0 * G * eps_e_dev


def get_lin_elas_vol(eps_e_vol, K):
    return K * eps_e_vol


def get_mu_I(I, mu_s, mu_d, I0):
    return mu_s + (mu_d - mu_s) * (1 / (1 + I0 / I))


def get_symmetric_part(A):
    return 0.5 * (A + A.T)


def get_skew_part(A):
    return 0.5 * (A - A.T)


def get_flattened_triu(A):
    return A.at[jnp.triu_indices(A.shape[0])].get()


def get_timestep(cell_size, bulk_modulus, shear_modulus, density, factor=0.1):
    c = jnp.sqrt((bulk_modulus + shear_modulus * (4.0 / 3.0)) / density)
    dt = factor * cell_size / c
    return dt


def get_symmetric_tensor_from_flattened_triu(vals, size=3):
    new = jnp.zeros((size, size))
    inds = jnp.triu_indices_from(new)
    new = new.at[inds].set(vals)
    new = new.at[inds[1], inds[0]].set(vals)

    return new

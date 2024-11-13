import operator
from inspect import signature

import jax
import jax.numpy as jnp


def get_sv(func, val):
    return signature(func).parameters[val].default


def get_dirpath():
    import os
    import sys

    file = sys.argv[0]
    return os.path.dirname(file) + "/"


def simple_warning(condition: jnp.bool_, place: str, message: str):
    jax.lax.cond(
        condition,
        lambda: jax.debug.print("{}: {}", place, message),
        lambda: None,
    )


def set_default_gpu(gpu_id=0):
    jax.config.update("jax_default_device", jax.devices("gpu")[gpu_id])

def dump_restart_files(
    config,
    solver=None,
    particles=None,
    nodes=None,
    material_stack=None,
    forces_stack=None,
    suffix="",
    directory=None,
):
    import pickle

    if directory is None:
        directory = "/restart"

    def dump(object, name):
        with open(
            f"{config.dir_path}/restart/{config.project}/{name}-{suffix}.pickle", "wb"
        ) as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dump(config, "config")

    if solver is not None:
        dump(solver, "solver")

    if particles is not None:
        dump(particles, "particles")

    if nodes is not None:
        dump(nodes, "nodes")

    if material_stack is not None:
        dump(material_stack, "material_stack")

    if forces_stack is not None:
        dump(forces_stack, "forces_stack")


def scan_kth(f, init, xs=None, reverse=False, unroll=1, store_every=1):
    """https://github.com/google/jax/discussions/12157"""
    store_every = operator.index(store_every)
    assert store_every > 0

    kwds = dict(reverse=reverse, unroll=unroll)

    if store_every == 1:
        return jax.lax.scan(f, init, xs=xs, **kwds)

    N, rem = divmod(len(xs), store_every)

    if rem:
        raise ValueError("store_every must evenly divide len(xs)")

    xs = xs.reshape(N, store_every, *xs.shape[1:])

    def f_outer(carry, xs):
        carry, ys = jax.lax.scan(f, carry, xs=xs, **kwds)
        jax.debug.print("step {} \r", xs[-1])
        return carry, [yss[-1] for yss in ys]

    return jax.lax.scan(f_outer, init, xs=xs, **kwds)

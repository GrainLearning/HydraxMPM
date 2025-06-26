# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import operator
from inspect import signature

import jax
import jax.numpy as jnp
import pickle


def get_sv(func, val):
    return signature(func).parameters[val].default


def get_dirpath():
    import os
    import sys

    file = sys.argv[0]
    return os.path.dirname(file) + "/"


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
        return carry, [yss[-1] for yss in ys]

    return jax.lax.scan(f_outer, init, xs=xs, **kwds)


def save_debug_file(debug_args, filename, where, why):
    """Saves material_points as a pickle file and returns it unchanged."""
    jax.debug.print("Debug triggered: '{}', in {} saving to {}", why, where, filename)
    # if filename is None:
    #     raise RuntimeError("filename not set cannot save debug files")

    with open(filename, "wb") as f:
        pickle.dump(debug_args, f)

    raise RuntimeError(f"Debug triggered in {where}, saved package to {filename}")


def debug_state(cond, args, filename="debug_package.pkl", where="...", why="..."):
    jax.lax.cond(
        cond,
        lambda _: jax.debug.callback(
            save_debug_file, debug_args=_, filename=filename, where=where, why=why
        ),
        lambda _: None,
        args,
    )

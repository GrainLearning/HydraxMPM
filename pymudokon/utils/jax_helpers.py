import jax

import operator


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

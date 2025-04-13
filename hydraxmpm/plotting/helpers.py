# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-


def make_plot(
    ax,
    x,
    y,
    start_end_markers=True,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    xlogscale=False,
    ylogscale=False,
    xmargin=None,
    ymargin=None,
    label=None,
    start_end_markersize=None,
    return_start_end_markers=False,
    **kwargs,
):
    import matplotlib.pyplot as plt

    from matplotlib.ticker import ScalarFormatter

    from matplotlib import ticker

    out = ax.plot(x, y, label=label, **kwargs)
    (line,) = out
    if start_end_markers:
        out_ms = ax.plot(
            x[0], y[0], ".", color=line.get_color(), markersize=start_end_markersize
        )
        out_me = ax.plot(
            x[-1], y[-1], "x", color=line.get_color(), markersize=start_end_markersize
        )

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    def format_func(value, tick_number):
        return "%g" % (value)

    if xlogscale:
        ax.set_xscale("log")

        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
    if ylogscale:
        ax.set_yscale("log")

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if xmargin:
        ax.margins(x=xmargin)
    if ymargin:
        ax.margins(y=ymargin)

    if return_start_end_markers:
        return out, out_ms, out_me
    else:
        return out

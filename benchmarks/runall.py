#! /usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import numpy as np
from matplotlib import pyplot
from numba.utils import benchmark

BENCHMARK_PREFIX = 'bm_'


def discover_files(startdir=os.curdir):
    for root, dirs, files in os.walk(startdir):
        for path in files:
            if path.startswith(BENCHMARK_PREFIX):
                fullpath = os.path.join(root, path)
                yield fullpath


try:
    from importlib import import_module
except ImportError:
    # Approximative fallback for Python < 2.7
    def import_module(modulename):
        module = __import__(modulename)
        for comp in modulename.split('.')[:-1]:
            module = getattr(module, comp)
        return module


def discover_modules():
    for fullpath in discover_files():
        path = os.path.relpath(fullpath)
        root, ext = os.path.splitext(path)
        if ext != '.py':
            continue
        modulename = root.replace(os.path.sep, '.')
        yield import_module(modulename)


def discover():
    for m in discover_modules():
        yield m.main


def run(mod):

    name = mod.__name__[len(BENCHMARK_PREFIX):]
    print('running', name, end=' ...\n')

    bmr = benchmark(mod.python_main)
    python_best = bmr.best
    print('\tpython', python_best, 'seconds')

    bmr = benchmark(mod.numba_main)
    numba_best = bmr.best
    print('\tnumba', numba_best, 'seconds')

    print('\tspeedup', python_best / numba_best)

    return name, numba_best / python_best


def main():
    # Generate timings
    labels = []
    scores = []
    for mod in discover_modules():
        label, result = run(mod)
        labels.append(label)
        scores.append(result)

    # Plot
    width = 0.8
    ind = np.arange(len(labels))
    fig, ax = pyplot.subplots()

    ax.bar(ind, scores, width)

    # Draw horizontal line at y=1
    ax.axhline(y=1, xmax=ind[-1], color='r')

    ax.set_ylabel('Normalized to CPython')
    ax.set_title('Numba Benchmark')
    ax.set_xticks(ind + (width/2))
    ax.set_xticklabels(labels)

    pyplot.show()

if __name__ == '__main__':
    main()

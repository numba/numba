"""
Modified from https://github.com/wesm/pandas/blob/master/vb_suite/suite.py
and https://github.com/wesm/pandas/blob/master/vb_suite/run_suite.py
"""

import os
import getpass
import sys
from datetime import datetime

from vbench.api import BenchmarkRunner, Benchmark, GitRepo

modules = [
    'mandelbrot',
    'filter',
]

by_module = {}
benchmarks = []

for modname in modules:
    ref = __import__(modname)
    by_module[modname] = [v for v in ref.__dict__.values()
                          if isinstance(v, Benchmark)]
    benchmarks.extend(by_module[modname])

for bm in benchmarks:
    assert(bm.name is not None)



USERNAME = getpass.getuser()

if sys.platform == 'darwin':
    HOME = '/Users/%s' % USERNAME
else:
    HOME = '/home/%s' % USERNAME

vb_suite_dir = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(vb_suite_dir)
REPO_URL = "git@github.com:numba/numba.git"
DB_PATH = os.path.join(vb_suite_dir, "benchmarks.db")
TMP_DIR = os.path.join(vb_suite_dir, "output")

PREPARE = """
python setup.py clean
"""
BUILD = """
git submodule init
git submodule update
python setup.py build_ext --inplace
"""
dependencies = ['pandas_vb_common.py']

START_DATE = datetime(2012, 11, 11)

repo = GitRepo(REPO_PATH)

RST_BASE = os.path.join(vb_suite_dir, 'source')

# HACK!

#timespan = [datetime(2011, 1, 1), datetime(2012, 1, 1)]

def generate_rst_files(benchmarks):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    vb_path = os.path.join(RST_BASE, 'vbench')
    fig_base_path = os.path.join(vb_path, 'figures')

    if not os.path.exists(vb_path):
        print 'creating %s' % vb_path
        os.makedirs(vb_path)

    if not os.path.exists(fig_base_path):
        print 'creating %s' % fig_base_path
        os.makedirs(fig_base_path)

    for bmk in benchmarks:
        print 'Generating rst file for %s' % bmk.name
        rst_path = os.path.join(RST_BASE, 'vbench/%s.txt' % bmk.name)

        fig_full_path = os.path.join(fig_base_path, '%s.png' % bmk.name)

        # make the figure
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        bmk.plot(DB_PATH, ax=ax)

        start, end = ax.get_xlim()

        plt.xlim([start - 30, end + 30])
        plt.savefig(fig_full_path, bbox_inches='tight')
        plt.close('all')

        fig_rel_path = 'vbench/figures/%s.png' % bmk.name
        rst_text = bmk.to_rst(image_path=fig_rel_path)
        with open(rst_path, 'w') as f:
            f.write(rst_text)

    with open(os.path.join(RST_BASE, 'index.rst'), 'w') as f:
        print >> f, """
Performance Benchmarks
======================

These historical benchmark graphs were produced with `vbench
<http://github.com/pydata/vbench>`__.

The ``pandas_vb_common`` setup script can be found here_

.. _here: https://github.com/pydata/pandas/tree/master/vb_suite

.. toctree::
    :hidden:
    :maxdepth: 3
"""
        for modname, mod_bmks in sorted(by_module.items()):
            print >> f, '    vb_%s' % modname
            modpath = os.path.join(RST_BASE, 'vb_%s.rst' % modname)
            with open(modpath, 'w') as mh:
                header = '%s\n%s\n\n' % (modname, '=' * len(modname))
                print >> mh, header

                for bmk in mod_bmks:
                    print >> mh, bmk.name
                    print >> mh, '-' * len(bmk.name)
                    print >> mh, '.. include:: vbench/%s.txt\n' % bmk.name

def run_process():
    runner = BenchmarkRunner(benchmarks, REPO_PATH, REPO_URL,
                             BUILD, DB_PATH, TMP_DIR, PREPARE,
                             always_clean=True,
                             run_option='eod', start_date=START_DATE,
                             module_dependencies=dependencies)
    runner.run()

if __name__ == '__main__':
    run_process()
    generate_rst_files(benchmarks)

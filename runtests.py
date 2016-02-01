#!/usr/bin/env python

import runpy

if __name__ == "__main__":
    runpy.run_module('numba.runtests', run_name='__main__')

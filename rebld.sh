rm build/*/numba/npyufunc/tbbpool.* build/*/numba/npyufunc/*.so numba/npyufunc/workqueue.*.so
python setup.py build_ext --inplace
NUMBA_USE_PARFOR=1 python ~/projects/BlackScholes_bench/bs_erf_numba_jit_par.py --steps=7

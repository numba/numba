
call activate %CONDA_ENV%

@rem Build numba extensions without silencing compile errors
python setup.py build_ext -q --inplace

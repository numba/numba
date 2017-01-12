
call activate %CONDA_ENV%

@rem Build numba extensions without silencing compile errors
python setup.py build_ext -q --inplace

@rem Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install -e .

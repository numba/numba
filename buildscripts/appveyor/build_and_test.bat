@rem This script will run in a single cmd environment, allowing to activate
@rem the environment

conda config --add channels numba
conda create -q -n testenv python=%CONDA_PY% numpy=%CONDA_NPY% llvmlite cffi scipy
call activate testenv
@echo on
%CMD_IN_ENV% python setup.py build_ext -q --inplace

@rem Run a subset of the test suite, as AppVeyor is quite slow
@rem (Python 3.4+ only)
@rem Also, note %CMD_IN_ENV% is needed for distutils/setuptools-based tests

%CMD_IN_ENV% python runtests.py -bv --random 0.1

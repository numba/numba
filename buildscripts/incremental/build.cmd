
call activate %CONDA_ENV%

@rem switch on the windows dll search mod
set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1


@rem Build numba extensions without silencing compile errors
python setup.py build_ext -q --inplace

@rem Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install -e .

if %errorlevel% neq 0 exit /b %errorlevel%

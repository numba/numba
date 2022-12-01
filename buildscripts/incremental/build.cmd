
call activate %CONDA_ENV%

git clone https://github.com/numba/llvmlite
pushd llvmlite
git fetch origin pull/869/head:pr/869
git checkout pr/869
python -m pip install . 
popd

@rem Build numba extensions without silencing compile errors
python setup.py build_ext -q --inplace

@rem Install numba locally for use in `numba -s` sys info tool at test time
python -m pip install -e .

if %errorlevel% neq 0 exit /b %errorlevel%

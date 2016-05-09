set CONDA_INSTALL=conda install -q -y
set PIP_INSTALL=pip install -q

conda update -q -y conda
@rem Clean up any left-over from a previous build
conda env remove -q -y -n %CONDA_ENV%
@rem Scipy, CFFI and jinja2 are optional dependencies, but exercised in the test suite
conda create -n %CONDA_ENV% -q -y python=%PYTHON% numpy=%NUMPY% cffi pip scipy jinja2
call activate %CONDA_ENV%
@rem Install llvmdev (separate channel, for now)
%CONDA_INSTALL% -c numba llvmdev="3.7*" llvmlite
@rem Install enum34 and singledispatch for Python < 3.4
if %PYTHON% LSS 3.4 (%CONDA_INSTALL% enum34)
if %PYTHON% LSS 3.4 (%PIP_INSTALL% singledispatch)
@rem Install funcsigs for Python < 3.3
if %PYTHON% LSS 3.3 (%CONDA_INSTALL% -c numba funcsigs)
@rem Install dependencies for building the documentation
if "%BUILD_DOC%" == "yes" (%CONDA_INSTALL% sphinx pygments)
if "%BUILD_DOC%" == "yes" (%PIP_INSTALL% sphinxjp.themecore sphinxjp.themes.basicstrap)
@rem Install dependencies for code coverage (codecov.io)
if "%RUN_COVERAGE%" == "yes" (%PIP_INSTALL% codecov)

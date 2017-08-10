@rem The cmd /C hack circumvents a regression where conda installs a conda.bat
@rem script in non-root environments.
set CONDA_INSTALL=cmd /C conda install -q -y
set PIP_INSTALL=pip install -q

@echo on

@rem Deactivate any environment
call deactivate
@rem Display root environment (for debugging)
conda list
@rem Clean up any left-over from a previous build
conda remove --all -q -y -n %CONDA_ENV%
@rem Scipy, CFFI, jinja2 and IPython are optional dependencies, but exercised in the test suite
conda create -n %CONDA_ENV% -q -y python=%PYTHON% numpy=%NUMPY% cffi pip scipy jinja2 ipython

call activate %CONDA_ENV%
@rem Install latest llvmlite build
%CONDA_INSTALL% -c numba llvmlite
@rem Install required backports for older Pythons
if %PYTHON% LSS 3.4 (%CONDA_INSTALL% enum34)
if %PYTHON% LSS 3.4 (%PIP_INSTALL% singledispatch)
if %PYTHON% LSS 3.3 (%CONDA_INSTALL% -c numba funcsigs)
@rem Install dependencies for building the documentation
if "%BUILD_DOC%" == "yes" (%CONDA_INSTALL% sphinx pygments)
if "%BUILD_DOC%" == "yes" (%PIP_INSTALL% sphinx_bootstrap_theme)
@rem Install dependencies for code coverage (codecov.io)
if "%RUN_COVERAGE%" == "yes" (%PIP_INSTALL% codecov)

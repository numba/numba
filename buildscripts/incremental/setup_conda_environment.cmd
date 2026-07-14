@rem first configure conda to have more tolerance of network problems, these
@rem numbers are not scientifically chosen, just merely larger than defaults
set CONDA_CONFIG=cmd /C conda config
%CONDA_CONFIG% --write-default
%CONDA_CONFIG% --set remote_connect_timeout_secs 30.15
%CONDA_CONFIG% --set remote_max_retries 10
%CONDA_CONFIG% --set remote_read_timeout_secs 120.2
%CONDA_CONFIG% --set show_channel_urls true
cmd /C conda info
%CONDA_CONFIG% --show

@rem The cmd /C hack circumvents a regression where conda installs a conda.bat
@rem script in non-root environments.
set CONDA_INSTALL=cmd /C conda install -q -y
set PIP_INSTALL=pip install -q

@echo on

@rem Deactivate any environment
call deactivate
@rem Display root environment (for debugging)
conda list
@rem Collect all conda packages so the environment is created in a single solve
@rem CFFI, jinja2 and IPython are optional dependencies, but exercised in the test suite
set PKGS=python=%PYTHON% numpy=%NUMPY% cffi pip jinja2 gitpython pyyaml psutil llvmlite=0.49 "tbb>=2021.6" "tbb-devel>=2021.6"
@rem missing IPython for Python 3.13
if "%PYTHON%" neq "3.13" set PKGS=%PKGS% ipython
@rem Install SciPy only if NumPy is not 2.1
if "%NUMPY%" neq "2.1" set PKGS=%PKGS% scipy
@rem Python 3.14+ requires setuptools
if "%PYTHON%" geq "3.14" set PKGS=%PKGS% "setuptools>=69.0.0"
@rem Install dependencies for building the documentation
if "%BUILD_DOC%" == "yes" set PKGS=%PKGS% sphinx sphinx_rtd_theme pygments
conda create -n %CONDA_ENV% -q -y -c numba/label/dev %PKGS%
if %errorlevel% neq 0 exit /b %errorlevel%

call activate %CONDA_ENV%
@rem Install dependencies for code coverage (codecov.io)
if "%RUN_COVERAGE%" == "yes" (%PIP_INSTALL% codecov)

echo "DEBUG ENV:"
echo "-------------------------------------------------------------------------"
conda env export
echo "-------------------------------------------------------------------------"

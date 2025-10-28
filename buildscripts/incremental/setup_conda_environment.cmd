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

@rem Install conda-anaconda-tos before creating environment to improve Azure CI detection
set CONDA_PLUGINS_AUTO_ACCEPT_TOS=true && %CONDA_INSTALL% "conda-anaconda-tos>=0.2.1" && set CONDA_PLUGINS_AUTO_ACCEPT_TOS=

if "%PYTHON%" neq "3.13" (
    @rem CFFI, jinja2 and IPython are optional dependencies, but exercised in the test suite
    conda create -n %CONDA_ENV% -q -y python=%PYTHON% numpy=%NUMPY% cffi pip jinja2 ipython gitpython pyyaml psutil
) else (
    @rem missing IPython for Python 3.13
    conda create -n %CONDA_ENV% -q -y python=%PYTHON% numpy=%NUMPY% cffi pip jinja2 gitpython pyyaml psutil
)
@rem Install SciPy only if NumPy is not 2.1
if "%NUMPY%" neq "2.1" (%CONDA_INSTALL% scipy)

call activate %CONDA_ENV%
@rem Install latest llvmlite build
%CONDA_INSTALL% -c numba/label/dev llvmlite=0.46
@rem Install dependencies for building the documentation
if "%BUILD_DOC%" == "yes" (%CONDA_INSTALL% sphinx sphinx_rtd_theme pygments)
@rem Install dependencies for code coverage (codecov.io)
if "%RUN_COVERAGE%" == "yes" (%PIP_INSTALL% codecov)
@rem Install TBB
%CONDA_INSTALL% "tbb>=2021.6" "tbb-devel>=2021.6"
if %errorlevel% neq 0 exit /b %errorlevel%

echo "DEBUG ENV:"
echo "-------------------------------------------------------------------------"
conda env export
echo "-------------------------------------------------------------------------"

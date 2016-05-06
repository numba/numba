
call activate %CONDA_ENV%
@rem Setup environment
set CONDA_INSTALL=conda install --yes -q
set PIP_INSTALL=pip install -q
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

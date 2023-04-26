
call activate %CONDA_ENV%

@rem Ensure that the documentation builds without warnings
if "%BUILD_DOC%" == "yes" python setup.py build_doc
@rem Run system info tool
pushd bin
numba -s
popd

@rem switch off color messages
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
@rem switch on developer mode
set NUMBA_DEVELOPER_MODE=1
@rem enable the faulthandler
set PYTHONFAULTHANDLER=1
@rem enable new style error handling
set NUMBA_CAPTURED_ERRORS=new_style

@rem First run Numba's Power-On-Self-Test to make sure testing will likely work
python -m numba.misc.POST
if %errorlevel% neq 0 exit /b %errorlevel%

@rem Now run the Numba test suite
@rem Note that coverage is run from the checkout dir to match the "source"
@rem directive in .coveragerc
if "%RUN_COVERAGE%" == "yes" (
    set PYTHONPATH=.
    coverage erase
    coverage run runtests.py -b --exclude-tags='long_running' -m -- numba.tests
) else (
    set NUMBA_ENABLE_CUDASIM=1
    python -m numba.runtests -b --exclude-tags='long_running' -m -- numba.tests
)

if %errorlevel% neq 0 exit /b %errorlevel%

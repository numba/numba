set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1

@rem Check Numba executables are there
pycc -h
numba -h

@rem Run system info tool
numba -s

@rem Check test discovery works
python -m numba.tests.test_runtests

@rem Run the whole test suite
python -m numba.runtests -b -m -- %TESTS_TO_RUN%

if errorlevel 1 exit 1

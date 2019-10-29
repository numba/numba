set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1
set TEST_NPROCS=4

@rem Check Numba executables are there
pycc -h
numba -h

@rem Run system info tool
numba -s

@rem Check test discovery works
@rem python -m numba.tests.test_runtests

@rem Run the whole test suite
python runtests.py -b --exclude-tags='long_running' -m 4 -- %TESTS_TO_RUN%

if errorlevel 1 exit 1

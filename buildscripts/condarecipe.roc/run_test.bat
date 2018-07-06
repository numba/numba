set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1

@rem Check Numba executables are there
pycc -h
numba -h

@rem Run system info tool
numba -s

@rem Check test discovery works
python -m numba.tests.test_runtests

@rem Run the ROC test suite
python -m numba.runtests -v -m -b numba.roc.tests

if errorlevel 1 exit 1

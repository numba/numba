set NUMBA_DEVELOPER_MODE=1
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
set PYTHONFAULTHANDLER=1
@rem Set CPU to generic to avoid LLVM 15 code bloat issue
set NUMBA_CPU_NAME=generic
set _NUMBA_REDUCED_TESTING=1

@rem Disable NumPy dispatching to AVX512_SKX feature extensions if the chip is
@rem reported to support the feature and NumPy >= 1.22 as this results in the use
@rem of low accuracy SVML libm replacements in ufunc loops.
set "_NPY_CMD=from numba.misc import numba_sysinfo;sysinfo=numba_sysinfo.get_sysinfo();print(sysinfo['NumPy AVX512_SKX detected'] and sysinfo['NumPy Version']^>='1.22')"
for /f %%i in ('python -c %_NPY_CMD%') do set NUMPY_DETECTS_AVX512_SKX_NP_GT_122=%%i
echo NumPy ^>= 1.22 with AVX512_SKX detected: %NUMPY_DETECTS_AVX512_SKX_NP_GT_122%

if "%NUMPY_DETECTS_AVX512_SKX_NP_GT_122%"=="True" (
    set NPY_DISABLE_CPU_FEATURES=AVX512_SKX
)

@rem Check Numba executable is there
numba -h

@rem Run system info tool
numba -s

@rem Check test discovery works
python -m numba.tests.test_runtests

@rem Run the whole test suite
python -m numba.runtests -b -m -- %TESTS_TO_RUN%

if errorlevel 1 exit 1

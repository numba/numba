
call activate %CONDA_ENV%

@rem Ensure that the documentation builds without warnings
if "%BUILD_DOC%" == "yes" python setup.py build_doc
@rem Run system info tool
pushd bin
numba -s
popd
@rem First check that the test discovery works
python -m numba.tests.test_runtests
@rem Now run the Numba test suite
@rem Note that coverage is run from the checkout dir to match the "source"
@rem directive in .coveragerc
if "%RUN_COVERAGE%" == "yes" (
    set PYTHONPATH=.
    coverage erase
    coverage run runtests.py -b -m numba.tests
) else (
    set NUMBA_ENABLE_CUDASIM=1
    python -m numba.runtests -b -m numba.tests
)

%PYTHON% setup.py install
if errorlevel 1 exit 1

rd /s /q %SP_DIR%\cython_debug

if "%PY3K%"=="1" (
    rd /s /q %SP_DIR%\numpy
)

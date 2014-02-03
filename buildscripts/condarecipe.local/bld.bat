xcopy %RECIPE_DIR%\..\.. src /s
cd src
%PYTHON% setup.py install
if errorlevel 1 exit 1

if "%PY3K%"=="1" (
    rd /s /q %SP_DIR%\numpy
)

REM First, setup the VS2022 environment
for /F "usebackq tokens=*" %%i in (`vswhere.exe -nologo -products * -version "[17.0,18.0)" -property installationPath`) do (
  set "VSINSTALLDIR=%%i\\"
)
if not exist "%VSINSTALLDIR%" (
  echo "Could not find VS 2022"
  exit /B 1
)

echo "Found VS 2022 in %VSINSTALLDIR%"
call "%VSINSTALLDIR%VC\\Auxiliary\\Build\\vcvarsall.bat" x64

%PYTHON% setup.py build install --single-version-externally-managed --record=record.txt

exit /b %errorlevel%

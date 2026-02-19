REM Setup VS2022 environment manually to avoid conda vs2022 activation issues
REM on Windows Server 2025 (hardcoded toolset version mismatches)
for /F "usebackq tokens=*" %%i in (`vswhere.exe -nologo -products * -version "[17.0,18.0)" -property installationPath`) do (
  set "VSINSTALLDIR=%%i\\"
)
if not exist "%VSINSTALLDIR%" (
  echo "Could not find VS 2022"
  exit /B 1
)
call "%VSINSTALLDIR%VC\Auxiliary\Build\vcvarsall.bat" x64

set NUMBA_PACKAGE_FORMAT=conda

%PYTHON% setup.py build install --single-version-externally-managed --record=record.txt

exit /b %errorlevel%

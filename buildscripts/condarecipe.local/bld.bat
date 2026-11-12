REM Setup VS environment (VS2022 or VS2026) manually to avoid conda vs2022
REM activation issues on Windows Server 2025
for /F "usebackq tokens=*" %%i in (`vswhere.exe -nologo -products * -version "[17.0,19.0)" -property installationPath`) do (
  set "VSINSTALLDIR=%%i\\"
)
if not exist "%VSINSTALLDIR%" (
  echo "Could not find VS 2022 or VS 2026"
  exit /B 1
)

REM Pin MSVC v14.44 (v143) to keep the vs2015_runtime >=14.44 run-dependency
REM pins valid; VS2026 would otherwise default to its native v14.5x toolset.
if "%ARCH%"=="arm64" (
  call "%VSINSTALLDIR%VC\Auxiliary\Build\vcvarsall.bat" ARM64 -vcvars_ver=14.44
) else (
  call "%VSINSTALLDIR%VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
)

%PYTHON% setup.py build install --single-version-externally-managed --record=record.txt

exit /b %errorlevel%

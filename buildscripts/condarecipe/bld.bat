set MVC=numbapro\vectorizers\workqueue.c



%PYTHON% setup.py install
if errorlevel 1 exit 1

rd /s /q %SP_DIR%\retired_cu
if errorlevel 1 exit 1

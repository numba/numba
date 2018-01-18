%PYTHON% buildscripts/remove_unwanted_files.py
%PYTHON% setup.py build install --single-version-externally-managed --record=record.txt

exit /b %errorlevel%

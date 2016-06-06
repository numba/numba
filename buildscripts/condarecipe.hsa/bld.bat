%PYTHON% buildscripts/remove_unwanted_files.py
%PYTHON% setup.py build install

exit /b %errorlevel%

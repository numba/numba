cd %RECIPE_DIR%\..
python build.py

mkdir %PREFIX%\DLLs
copy %RECIPE_DIR%\..\lib\* %PREFIX%\DLLs

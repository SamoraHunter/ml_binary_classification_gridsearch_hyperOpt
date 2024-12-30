@echo off

rem Check if Python is installed and on path
where python >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_COMMAND=python
) else (
    where python3 >nul 2>nul
    if %errorlevel% equ 0 (
        set PYTHON_COMMAND=python3
    ) else (
        echo Python is not installed or not on path.
        exit /b
    )
)

echo %PYTHON_COMMAND% is installed and on path.

rem Check if "ml_grid_env" folder exists in the current directory
if exist "ml_grid_env" (
    echo ml_grid_env folder exists in the current directory.
) else (
    echo ml_grid_env folder does not exist in the current directory.
    echo Creating ml_grid_env folder...
    %PYTHON_COMMAND% -m venv ml_grid_env
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Exiting...
        exit /b
    )
    echo ml_grid_env folder created.
)

rem Activate the virtual environment
call ml_grid_env\Scripts\activate.bat

if errorlevel 1 (
    echo Warning: Failed to activate virtual environment.
    exit /b
)

echo Virtual environment activated successfully.

rem Upgrade pip
echo Upgrading pip...
call ml_grid_env\Scripts\python.exe -m pip install --upgrade pip

rem Install requirements 
echo Installing requirements...
pip install -r requirements.txt

rem Install ipykernel and add ml_grid_env to the kernel spec
echo Installing ipykernel...
pip install ipykernel

echo Adding ml_grid_env to the kernel spec...
call ml_grid_env\Scripts\python.exe -m ipykernel install --user --name=ml_grid_env

rem Deactivate the virtual environment
echo Deactivating virtual environment...
deactivate

echo Virtual environment deactivated.

echo All operations completed.
exit /b

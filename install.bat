@echo off

rem Function to print error message and exit
:print_error_and_exit
echo %1
echo Please try deleting the existing 'ml_grid_env' directory and run the script again.
exit /b 1

rem Check if Python 3 is found
where python >nul 2>nul
if errorlevel 1 (
    call :print_error_and_exit "Python 3 is not found. Please make sure Python 3 is installed."
)

rem Check if virtual environment exists
if not exist "ml_grid_env" (
    rem Create virtual environment
    python -m venv ml_grid_env || call :print_error_and_exit "Failed to create virtual environment"
)

rem Activate virtual environment
call ml_grid_env\Scripts\activate || call :print_error_and_exit "Failed to activate virtual environment"

rem Upgrade pip
python -m pip install --upgrade pip

rem Install ipykernel
pip install ipykernel

rem Add kernel spec
python -m ipykernel install --user --name=ml_grid_env

rem Install requirements from requirements.txt
for /f "delims=" %%a in (requirements.txt) do (
    pip install %%a
    if not errorlevel 0 (
        echo Failed to install %%a >> installation_log.txt
    ) else (
        echo Successfully installed %%a
    )
)

rem Deactivate virtual environment
deactivate

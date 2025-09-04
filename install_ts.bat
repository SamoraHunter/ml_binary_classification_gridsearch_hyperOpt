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

rem Check if virtual environment exists
if exist "ml_grid_ts_env" (
    echo ml_grid_ts_env folder exists in the current directory.
) else (
    echo ml_grid_ts_env folder does not exist in the current directory.
    echo Creating ml_grid_ts_env folder...
    rem Create virtual environment
    %PYTHON_COMMAND% -m venv ml_grid_ts_env
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Exiting...
        exit /b
    )
    echo ml_grid_ts_env folder created.
)

rem Activate virtual environment
echo Activating virtual environment...
call ml_grid_ts_env\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment. Exiting.
    exit /b
)
echo Virtual environment activated.

rem Upgrade pip
echo Upgrading pip...
call ml_grid_ts_env\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip. Exiting.
    exit /b
)

rem Install the project in editable mode with time-series and testing dependencies
echo Installing project dependencies...
call ml_grid_ts_env\Scripts\python.exe -m pip install -e .[test,ts]
if errorlevel 1 (
    echo ERROR: Failed to install project dependencies. Exiting.
    exit /b
)

rem Add kernel spec
echo Adding Jupyter kernel spec...
call ml_grid_ts_env\Scripts\python.exe -m ipykernel install --user --name=ml_grid_ts_env

rem Deactivate virtual environment
echo Deactivating virtual environment...
deactivate

echo All operations completed.
exit /b

@echo off
setlocal

rem Default values for standard installation
set "ENV_NAME=ml_grid_env"
set "EXTRAS=[test]"
set "INSTALL_TYPE=standard"

rem Check for 'ts' argument for time-series installation
if /i "%1" == "ts" (
    set "ENV_NAME=ml_grid_ts_env"
    set "EXTRAS=[test,ts]"
    set "INSTALL_TYPE=time-series"
)

echo Starting %INSTALL_TYPE% installation...
echo Environment will be created at: .\%ENV_NAME%

rem Function-like behavior with goto
:error
echo ERROR: %~1
echo Installation failed. Please try deleting the existing ".\%ENV_NAME%" directory and run the script again.
exit /b 1

rem Check if Python is installed and on path
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
) else (
    where python3 >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=python3"
    ) else (
        call :error "Python is not installed or not on path."
    )
)

rem Check if "ml_grid_env" folder exists in the current directory
if not exist "%ENV_NAME%\" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv "%ENV_NAME%"
    if %errorlevel% neq 0 (
        call :error "Failed to create virtual environment."
    )
)

rem Activate the virtual environment
call "%ENV_NAME%\Scripts\activate.bat" || call :error "Failed to activate virtual environment."
echo Virtual environment activated successfully.

rem Upgrade pip
echo Upgrading pip...
pip install --upgrade pip setuptools wheel

rem Install the project in editable mode along with testing dependencies.
rem This reads all dependencies from pyproject.toml.
echo Installing project dependencies (%EXTRAS%)...
pip install -e .%EXTRAS%
if %errorlevel% neq 0 (
    call :error "Failed to install project dependencies."
)

rem Add kernel spec for Jupyter
echo Registering Jupyter kernel...
python -m ipykernel install --user --name="%ENV_NAME%" --display-name="Python (%ENV_NAME%)"

rem Deactivate the virtual environment
echo Deactivating virtual environment...
deactivate
echo Virtual environment deactivated.

echo.
echo [SUCCESS] Installation complete!
echo To activate the '%INSTALL_TYPE%' environment, run:
echo .\%ENV_NAME%\Scripts\activate.bat

endlocal
exit /b 0

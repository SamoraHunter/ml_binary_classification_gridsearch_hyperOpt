@echo off

rem Check if virtual environment exists
if not exist ml_grid_ts_env (
    rem Create virtual environment
    python -m venv ml_grid_ts_env
)

rem Activate virtual environment
call ml_grid_ts_env\Scripts\activate

rem Upgrade pip
python -m pip install --upgrade pip

rem Install ipykernel
pip install ipykernel

rem Add kernel spec
python -m ipykernel install --user --name=ml_grid_ts_env

rem Install requirements from requirements.txt
for /f "delims=" %%i in (requirements.txt) do (
    pip install %%i
    if errorlevel 1 (
        echo Failed to install %%i >> installation_log.txt
    ) else (
        echo Successfully installed %%i
    )
)

rem Install requirements from requirements_ts.txt
for /f "delims=" %%i in (requirements_ts.txt) do (
    pip install %%i
    if errorlevel 1 (
        echo Failed to install %%i >> installation_log.txt
    ) else (
        echo Successfully installed %%i
    )
)

rem Deactivate virtual environment
deactivate

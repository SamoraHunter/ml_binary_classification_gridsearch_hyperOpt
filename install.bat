@echo off

REM Create virtual environment
python -m venv ml_grid_ts_env
call ml_grid_ts_env\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install IPython kernel spec
python -m pip install ipykernel

python -m ipykernel install --user --name=ml_grid_ts_env

REM Install packages from requirements.txt
python -m pip install -r requirements.txt || (
    echo Failed to install some packages from requirements.txt
)

REM Install packages from requirements_ts.txt
python -m pip install -r requirements_ts.txt || (
    echo Failed to install some packages from requirements_ts.txt
)

echo Installation completed successfully.
pause

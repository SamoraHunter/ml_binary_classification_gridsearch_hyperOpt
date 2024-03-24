#!/bin/bash

# Check if virtual environment exists
if [ ! -d "ml_grid_ts_env" ]; then
    # Create virtual environment
    python3 -m venv ml_grid_ts_env
fi

# Activate virtual environment
source ml_grid_ts_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install ipykernel
pip install ipykernel

# Add kernel spec
python -m ipykernel install --user --name=ml_grid_ts_env

# Install requirements from requirements.txt
while read -r package; do
    pip install "$package"
    if [ $? -ne 0 ]; then
        echo "Failed to install $package" >> installation_log.txt
    else
        echo "Successfully installed $package"
    fi
done < requirements.txt

# Install requirements from requirements_ts.txt
while read -r package; do
    pip install "$package"
    if [ $? -ne 0 ]; then
        echo "Failed to install $package" >> installation_log.txt
    else
        echo "Successfully installed $package"
    fi
done < requirements_ts.txt

# Deactivate virtual environment
deactivate

#!/bin/bash

# Function to print error message and exit
print_error_and_exit() {
    echo "$1"
    echo "Please try deleting the existing 'ml_grid_env' directory and run the script again."
    exit 1
}

# Check if virtual environment exists
if [ ! -d "ml_grid_env" ]; then
    # Create virtual environment
    python -m venv ml_grid_env || print_error_and_exit "Failed to create virtual environment"
fi

# Activate virtual environment
source ml_grid_env/bin/activate || print_error_and_exit "Failed to activate virtual environment"

# Upgrade pip
python -m pip install --upgrade pip

# Install ipykernel
pip install ipykernel

# Add kernel spec
python -m ipykernel install --user --name=ml_grid_env

# Install requirements from requirements.txt
pip install -r requirements.txt

# Deactivate virtual environment
deactivate
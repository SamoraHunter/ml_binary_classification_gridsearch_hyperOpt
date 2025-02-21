#!/bin/bash

# Function to print error message and exit
print_error_and_exit() {
    echo "$1"
    echo "Please try deleting the existing 'ml_grid_env' directory and run the script again."
    exit 1
}

# Determine whether to use python or python3
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    print_error_and_exit "Python is not installed. Please install Python and try again."
fi

# Check if virtual environment exists
if [ ! -d "ml_grid_env" ]; then
    # Create virtual environment
    $PYTHON_CMD -m venv ml_grid_env || print_error_and_exit "Failed to create virtual environment"
fi

# Activate virtual environment
source ml_grid_env/bin/activate || print_error_and_exit "Failed to activate virtual environment"

# Upgrade pip
$PYTHON_CMD -m pip install --upgrade pip

# Install ipykernel
pip install ipykernel

# Add kernel spec
$PYTHON_CMD -m ipykernel install --user --name=ml_grid_env

# Install requirements from requirements.txt
pip install -r requirements.txt

# Deactivate virtual environment
deactivate

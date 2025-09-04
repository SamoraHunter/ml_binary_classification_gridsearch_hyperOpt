#!/bin/bash

# Default values for standard installation
ENV_NAME="ml_grid_env"
EXTRAS="[test]"
INSTALL_TYPE="standard"

# Check for 'ts' argument for time-series installation
if [ "$1" == "ts" ]; then
    ENV_NAME="ml_grid_ts_env"
    EXTRAS="[test,ts]"
    INSTALL_TYPE="time-series"
fi

# Function to print error message and exit
print_error_and_exit() {
    echo "ERROR: $1"
    echo "Installation failed. Please try deleting the existing './$ENV_NAME' directory and run the script again."
    exit 1
}

# Determine whether to use python or python3
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    print_error_and_exit "Python is not installed. Please install it and try again."
fi

# Check if virtual environment exists
if [ ! -d "$ENV_NAME" ]; then
    # Create virtual environment
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$ENV_NAME" || print_error_and_exit "Failed to create virtual environment."
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "$ENV_NAME/bin/activate" || print_error_and_exit "Failed to activate virtual environment"

echo "Virtual environment activated."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the project in editable mode along with testing dependencies.
# This reads all dependencies from pyproject.toml.
echo "Installing project dependencies ($EXTRAS)..."
pip install -e ."$EXTRAS" || print_error_and_exit "Failed to install project dependencies."

# Add kernel spec for Jupyter
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name="$ENV_NAME" --display-name="Python ($ENV_NAME)"

# Deactivate virtual environment
deactivate
echo "Virtual environment deactivated."

echo ""
echo "✅ Installation complete!"
echo "To activate the '$INSTALL_TYPE' environment, run:"
echo "source $ENV_NAME/bin/activate"

#!/bin/bash

# Default values for standard installation
ENV_NAME="ml_grid_env"
EXTRAS="[test,standard]"
INSTALL_TYPE="standard"
PROXY_MODE=false

# Function to print error message and exit
print_error_and_exit() {
    echo "ERROR: $1"
    echo "Installation failed. Please try deleting the existing './$ENV_NAME' directory and run the script again."
    exit 1
}

show_help() {
    echo "Usage: ./install_ml_grid.sh [OPTIONS] [ts]"
    echo ""
    echo "Arguments:"
    echo "  ts                   Install time-series variant (ml_grid_ts_env with [test,ts] extras)"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -p, --proxy          Install with proxy support"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--proxy) PROXY_MODE=true; shift;;
        -h|--help) show_help; exit 0;;
        ts)
            ENV_NAME="ml_grid_ts_env"
            EXTRAS="[test,ts]"
            INSTALL_TYPE="time-series"
            shift;;
        *) echo "Unknown option: $1"; show_help; exit 1;;
    esac
done

echo "Detecting Python interpreter..."

if command -v python3.10 &>/dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.11 &>/dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    print_error_and_exit "Python is not installed."
fi

# Debug output
PYTHON_PATH=$(command -v $PYTHON_CMD)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)

echo "Using Python command: $PYTHON_CMD"
echo "Resolved path: $PYTHON_PATH"
echo "Python version: $PYTHON_VERSION"
echo ""

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

echo "Upgrading pip, setuptools, and wheel..."
pip_base_args=()
if [ "$PROXY_MODE" = true ]; then
    pip_base_args+=("--trusted-host" "dh-cap02" "-i" "http://dh-cap02:8008/mirrors/ml_binary_classification_gridsearch_hyperOpt")
fi

python -m pip install --upgrade pip 'setuptools<70.0.0' wheel "${pip_base_args[@]}" \
    || print_error_and_exit "Failed to upgrade pip."

# Increase the resolver backtrack depth to handle the large, complex dependency
# graph (TensorFlow + PyTorch + AutoGluon + FastAI + aeon[all_extras] etc.).
# PIP_RESOLVER_MAX_BACKTRACK is honoured by pip >= 23.3.
export PIP_RESOLVER_MAX_BACKTRACK=1000000

# Pre-install heavy dependencies to simplify graph resolution
echo "Pre-installing heavy dependencies..."
python -m pip install "tensorflow==2.20.0" "torch==2.10.0" "nvidia-cuda-nvcc-cu12" "h2o>=3.46.0.5" "scikit-learn>=1.6.0,<1.7" "${pip_base_args[@]}" || print_error_and_exit "Failed to pre-install frameworks."

if [ "$INSTALL_TYPE" = "time-series" ]; then
    echo "Pre-installing time-series dependencies to simplify graph resolution..."
    CONSTRAINTS_FILE="ts-constraints.txt"

    # Pre-install the top-level heavy time-series libraries.
    # This allows pip to resolve their complex dependencies in a dedicated step
    # before the final project installation. These are the main libraries
    # from the '[project.optional-dependencies].ts' section in pyproject.toml.
    python -m pip install \
        "aeon>=1.2.0" \
        "tsfresh" \
        "prophet==1.1.3" \
        "pmdarima==2.0.3" \
        "gluonts>=0.14.0" \
        "${pip_base_args[@]}" || print_error_and_exit "Failed to install heavy time-series libraries."

    echo "Generating constraints file to pin pre-installed packages..."
    pip freeze > "$CONSTRAINTS_FILE"
fi

# Install the project in editable mode along with testing dependencies.
# This reads all dependencies from pyproject.toml.
echo "Installing project dependencies ($EXTRAS)..."
pip_install_args=("--no-build-isolation" "-e" ".$EXTRAS")
if [ "$INSTALL_TYPE" = "time-series" ]; then
    pip_install_args+=("-c" "$CONSTRAINTS_FILE")
fi
if [ "$PROXY_MODE" = true ]; then
    pip_install_args+=("--trusted-host" "dh-cap02" "-i" "http://dh-cap02:8008/mirrors/ml_binary_classification_gridsearch_hyperOpt" "--retries" "5" "--timeout" "60")
fi
pip install "${pip_install_args[@]}" || print_error_and_exit "Failed to install project dependencies."

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

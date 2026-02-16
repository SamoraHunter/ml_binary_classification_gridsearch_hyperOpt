#!/bin/bash
# .githooks/pre-push

# Navigate to the repository root (relative to the hook location)
cd "$(dirname "$0")/.."

# Activate the virtual environment (if needed)
VENV_PATH=$(find . -type d -name "ml_grid_env")
if [ -n "$VENV_PATH" ]; then
  source "$VENV_PATH/bin/activate"
else
  echo "Virtual environment not found. Exiting."
  exit 1
fi

# Run Ruff linter
echo "Running Ruff linter..."
ruff check .
if [ $? -ne 0 ]; then
    echo "Ruff linting failed. Please fix the issues before pushing."
    exit 1
fi

# Run Black formatter check
echo "Running Black formatter check..."
black --check .
if [ $? -ne 0 ]; then
    echo "Black formatting check failed. Please run 'black .' to format your code before pushing."
    exit 1
fi

# Run the same test command as in the GitHub Actions workflow
echo "Running tests before pushing..."
pytest notebooks/test_notebook.py

# Check the exit status of the test command
if [ $? -ne 0 ]; then
    echo "Tests failed. Push aborted."
    exit 1
fi

echo "All checks passed. Proceeding with push."
exit 0

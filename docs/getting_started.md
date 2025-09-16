# Getting Started

This guide will help you get the `ml-grid` project up and running.

## Installation

This project includes convenient installation scripts for Unix/Linux/macOS and Windows. These scripts will create a virtual environment, install all necessary dependencies, and register a Jupyter kernel for you.

1. **Clone the repository:**
    ```shell
    git clone https://github.com/SamoraHunter/ml_binary_classification_gridsearch_hyperOpt.git
    cd ml_binary_classification_gridsearch_hyperOpt
    ```

2.  **Run the installation script:**

    *   **For a standard installation:**
        *   On Unix/Linux/macOS:
            ```bash
            chmod +x install.sh
            ./install.sh
            ```
        *   On Windows:
            ```bat
            install.bat
            ```
        This will create a virtual environment named `ml_grid_env`.

## Usage

After installation, activate the virtual environment to run your code or notebooks.

*   **To activate the standard environment:**
    *   On Unix/Linux/macOS: `source ml_grid_env/bin/activate`
    *   On Windows: `.\ml_grid_env\Scripts\activate`
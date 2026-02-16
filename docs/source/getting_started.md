# Getting Started

This guide will help you get the `ml-grid` project up and running.

## Installation

This project includes convenient installation scripts for Unix/Linux/macOS and Windows. These scripts will create a virtual environment, install all necessary dependencies, and register a Jupyter kernel for you.

### Quick Install using Scripts

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

    *   **For a time-series installation (includes all standard dependencies):**
        *   On Unix/Linux/macOS:
            ```bash
            chmod +x install.sh
            ./install.sh ts
            ```
        *   On Windows:
            ```bat
            install.bat ts
            ```
        This will create a virtual environment named `ml_grid_ts_env`.

## Usage

After installation, activate the virtual environment to run your code or notebooks.

*   **To activate the standard environment:**
    *   On Unix/Linux/macOS: `source ml_grid_env/bin/activate`
    *   On Windows: `.\ml_grid_env\Scripts\activate`

*   **To activate the time-series environment:**
    *   On Unix/Linux/macOS: `source ml_grid_ts_env/bin/activate`
    *   On Windows: `.\ml_grid_ts_env\Scripts\activate`

If you are using Jupyter, you can also select the kernel created during installation (e.g., `Python (ml_grid_env)`) directly from the Jupyter interface.

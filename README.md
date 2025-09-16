# ml_binary_classification_gridsearch_hyperOpt

This repository contains Python code for binary classification using grid search and hyperparameter optimization techniques.

# Table of Contents

- [ml_binary_classification_gridsearch_hyperOpt](#ml_binary_classification_gridsearch_hyperopt)
- [Overview](#overview)
- [Diagrams](#diagrams)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Windows](#windows)
  - [Unix/Linux](#unixlinux)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Appendix](#appendix)
- [Acknowledgments](#acknowledgments)


## Overview

Binary classification is a common machine learning task where the goal is to categorize data into one of two classes. This repository provides a framework for performing binary classification using various machine learning algorithms and optimizing their hyperparameters through grid search and hyperparameter optimization techniques.

## Diagrams

Below are visual diagrams representing various components of the project. All `.mmd` source files are Mermaid diagrams, and the rendered versions are available in `.svg` or `.png` formats.

### Feature Importance
- [Mermaid source](assets/data_feature_importance_methods.mmd)  
  <img src="assets/data_feature_importance_methods.svg" width="400" height="300"/>

### Data Pipeline
- [Mermaid source](assets/data_pipeline.mmd)  
  <img src="assets/data_pipeline.svg" width="400" height="300"/>

### Grid Parameter Search Space
- [Mermaid source](assets/grid_param_space.mmd)  
  <img src="assets/grid_param_space.svg" width="400" height="300"/>

### Hyperparameter Search
- [Mermaid source](assets/hyperparameter_search.mmd)  
  <img src="assets/hyperparameter_search.svg" width="400" height="300"/>

### Imputation Pipeline
- [Mermaid source](assets/impute_data_for_pipe.mmd)  
  <img src="assets/impute_data_for_pipe.svg" width="400" height="300"/>

### ML Repository Architecture
- [Mermaid source](assets/ml_repository_architecture.mmd)  
  <img src="assets/ml_repository_architecture.png" width="400" height="300"/>

### Model Class Listing (Time Series)
- [Mermaid source](assets/model_class_list_model_class_list_ts.mmd)  
  <img src="assets/model_class_list_model_class_list_ts.svg" width="400" height="300"/>

### Project Scoring and Model Saving
- [Mermaid source](assets/project_score_save.mmd)  
  <img src="assets/project_score_save.svg" width="400" height="300"/>

### Time Series Helper Functions
- [Mermaid source](assets/time_series_helper.mmd)  
  <img src="assets/time_series_helper.svg" width="400" height="300"/>

### Unit Test - Synthetic Data
- [Mermaid source](assets/unit_test_synthetic.mmd)  
  <img src="assets/unit_test_synthetic.svg" width="400" height="300"/>

### Results Processing Pipeline
- [Mermaid source](assets/results_processing_pipeline.mmd)
  <img src="assets/results_processing_pipeline.svg" width="600" height="450"/>


## Getting Started

### Prerequisites

Designed for usage with a numeric data matrix for binary classification. Single or multiple outcome variables (One v rest). An example is provided. Time series is also implemented.

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

## Examples

See [ml_grid/tests/unit_test_synthetic.ipynb]

## Documentation

This project uses Sphinx for documentation. The documentation includes usage guides and an auto-generated API reference.

To build the documentation locally:

1.  Install the documentation dependencies (make sure your virtual environment is activated):
    ```bash
    pip install -e .[docs]
    ```

2.  Build the HTML documentation:
    ```bash
    sphinx-build -b html docs/source docs/build
    ```

3.  Open `docs/build/index.html` in your web browser to view the documentation.

## Contributing
If you would like to contribute to this project, please follow these steps:

Fork the repository on GitHub.
Create a new branch for your feature or bug fix.
Make your changes and commit them with descriptive commit messages.
Push your changes to your fork.
Create a pull request to the main repository's master branch.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Appendix


## Acknowledgments
scikit-learn
hyperopt

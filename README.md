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

### Windows:

1. **Clone the repository:**
    ```shell
    git clone https://github.com/SamoraHunter/ml_binary_classification_gridsearch_hyperOpt.git
    cd ml_binary_classification_gridsearch_hyperOpt
    ```

2. **Run the installation script:**
    ```shell
    install.bat
    ```

### Unix/Linux:

1. **Clone the repository:**
    ```shell
    git clone https://github.com/SamoraHunter/ml_binary_classification_gridsearch_hyperOpt.git
    cd ml_binary_classification_gridsearch_hyperOpt
    ```

2. **Run the installation script:**
    ```shell
    chmod +x install.sh
    ./install.sh
    ```


```python
import sys
sys.path.append('/path/to/ml_grid')
import ml_grid


```

## Usage

See Appendix

## Examples

See [ml_grid/tests/unit_test_synthetic.ipynb]


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

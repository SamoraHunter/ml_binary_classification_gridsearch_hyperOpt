# ml_binary_classification_gridsearch_hyperOpt

This repository contains Python code for binary classification using grid search and hyperparameter optimization techniques.

# Table of Contents

- [ml_binary_classification_gridsearch_hyperOpt](#ml_binary_classification_gridsearch_hyperopt)
- [Overview](#overview)
- [ML repository architecture](#ml-repository-architecture)
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

## ML repository architecture

![Alt text](assets/ml_repository_architecture.png)

## Getting Started

### Prerequisites

Before you can run the code in this repository, make sure you have the following prerequisites installed:

- Python (>=3.6) -requirements.txt built for python3.10.12
- NumPy
- Pandas
- Scikit-Learn
- HyperOpt (for hyperparameter optimization)
- Pytorch

You can install these dependencies using pip:

pip install numpy pandas scikit-learn hyperopt

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



Contributing
If you would like to contribute to this project, please follow these steps:

Fork the repository on GitHub.
Create a new branch for your feature or bug fix.
Make your changes and commit them with descriptive commit messages.
Push your changes to your fork.
Create a pull request to the main repository's master branch.
License
This project is licensed under the MIT License - see the LICENSE file for details.


## Appendix



Acknowledgments
scikit-learn
hyperopt

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random


def get_data_split(X, y, local_param_dict):
    """
    Split data into train, test, and validation sets
    based on user inputs in local_param_dict

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features.
    y : array-like of shape (n_samples,)
        Target variable.
    local_param_dict: dict
        Dictionary of user-defined parameters for data splitting

    Returns
    -------
    X_train : array-like of shape (n_samples_train, n_features)
        Features for training.
    X_test : array-like of shape (n_samples_test, n_features)
        Features for testing.
    y_train : array-like of shape (n_samples_train,)
        Target variable for training.
    y_test : array-like of shape (n_samples_test,)
        Target variable for testing.
    X_test_orig : array-like of shape (n_samples_test, n_features)
        Original features for validation.
    y_test_orig : array-like of shape (n_samples_test,)
        Original target variable for validation.
    """
    # X = X
    # y = y
    # local_param_dict = local_param_dict
    # X_train_orig, X_test_orig, y_train_orig, y_test_orig = None, None, None, None
    
    random.seed(1234)
    np.random.seed(1234)

    # Check if data is valid
    if not is_valid_shape(X):
        local_param_dict["resample"] = None
        print("overriding resample with None")

    # No resampling
    if local_param_dict.get("resample") is None:
        # Split into training and testing sets
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        # Split training set into final training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )

    # Undersampling
    elif local_param_dict.get("resample") == "undersample":
        # print("undersample..")
        # print((y.shape))
        # print(X.shape)

        # Undersample data
        rus = RandomUnderSampler(random_state=0)
        X, y = rus.fit_resample(X, y)

        # Split into training and testing sets
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        # Split training set into final training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )
        X = X_train_orig.copy()
        y = y_train_orig.copy()

    # Oversampling
    elif local_param_dict.get("resample") == "oversample":
        # Train test split
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        # Oversample training set
        sampling_strategy = 1
        ros = RandomOverSampler(sampling_strategy=sampling_strategy)
        X_train_orig, y_train_orig = ros.fit_resample(X_train_orig, y_train_orig)
        print(y_train_orig.value_counts())

        # Split training set into final training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )

    return X_train, X_test, y_train, y_test, X_test_orig, y_test_orig


# check names! Random resampling

#         X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
#         X, y, test_size=0.25, random_state=1
#         )

#         sampling_strategy = 0.8
#         ros = RandomOverSampler(sampling_strategy=sampling_strategy)
#         X_res, y_res = ros.fit_resample(X_train_orig, y_train_orig)
#         print(y_res.value_counts())
#         X = X_res.copy()
#         y = y_res.copy()

#         X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=1)


def is_valid_shape(input_data):
    # Check if input_data is a numpy array
    if isinstance(input_data, np.ndarray):
        # If it's a numpy array, directly check its number of dimensions
        return input_data.ndim == 2

    # Check if input_data is a pandas DataFrame
    elif isinstance(input_data, pd.DataFrame):
        # If it's a DataFrame, convert it to a numpy array and then check its shape
        input_array = input_data.values
        return input_array.ndim == 2

    else:
        # Input data is neither a numpy array nor a pandas DataFrame
        return False


## Reproduce data split:


# from ml_grid.pipeline.data_train_test_split import get_data_split

# local_param_dict  = {'resample': str(df.iloc[0]['resample'])}

# # replace nan value in local_param_dict with None

# local_param_dict = {k: v if v!= 'nan' else None for k, v in local_param_dict.items()}

# print(local_param_dict)


# X_train, X_test, y_train, y_test, X_train_orig, y_test_orig = get_data_split(X, y, local_param_dict)
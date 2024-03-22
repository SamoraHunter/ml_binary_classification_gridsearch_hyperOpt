from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def get_data_split(X, y, local_param_dict):

    # X = X
    # y = y
    # local_param_dict = local_param_dict
    # X_train_orig, X_test_orig, y_train_orig, y_test_orig = None, None, None, None

    if local_param_dict.get("resample") == None:

        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )

        # retain validation set with X_test_orig and y_test_orig

        # training sets are only X_train, X_test y_train y_test

    elif local_param_dict.get("resample") == "undersample":
        print("undersample..")
        print((y.shape))
        print(X.shape)
        rus = RandomUnderSampler(random_state=0)
        X, y = rus.fit_resample(X, y)
        # Create validation set
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        # X = X_train_orig.copy()
        # y = y_train_orig.copy()
        # Resplit holding back _orig
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )
        X = X_train_orig.copy()
        y = y_train_orig.copy()

    elif local_param_dict.get("resample") == "oversample":
        # Train test split then oversample to avoid poison
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        sampling_strategy = 1
        ros = RandomOverSampler(sampling_strategy=sampling_strategy)
        X_train_orig, y_train_orig = ros.fit_resample(X_train_orig, y_train_orig)
        print(y_train_orig.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X_train_orig, y_train_orig, test_size=0.25, random_state=1
        )

        # return training sets and final validation sets.

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

class RocketClassifier_class():
    algorithm_implementation = RocketClassifier

    method_name = 'RocketClassifier'

    parameter_space = {
        'num_kernels': [5000, 10000, 15000],  # The number of kernels for the Rocket transform.
        'rocket_transform': ["rocket", "minirocket", "multirocket"],  # The type of Rocket transformer to use. Valid inputs = ["rocket", "minirocket", "multirocket"].
        'max_dilations_per_kernel': [16, 32, 64],  # MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
        'n_features_per_kernel': [3, 4, 5],  # MultiRocket only. The number of features per kernel.
        'random_state': [random_state_val],  # Seed for random number generation.
        'estimator': [None],  # If none, a RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)) is used.
        'n_jobs': [n_jobs_model_val],  # Number of threads to use for the convolutional transform. -1 means using all processors.
    }
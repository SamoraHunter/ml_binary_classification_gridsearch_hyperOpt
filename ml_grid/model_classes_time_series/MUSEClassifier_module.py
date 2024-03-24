class MUSE_class():
    algorithm_implementation = MUSE

    method_name = 'MUSE'

    parameter_space = {
        'anova': [True, False],                         # If True, Fourier coefficient selection is done via a one-way ANOVA test
        'variance': [True, False],                      # If True, Fourier coefficient selection is done via the largest variance
        'bigrams': [True, False],                       # Whether to create bigrams of SFA words
        'window_inc': [2, 4],                           # Increment used to determine the next window size for BoP model
        'alphabet_size': [4, 6, 8],                     # Number of possible letters (values) for each word
        'use_first_order_differences': [True, False],   # If True, adds the first order differences of each dimension to the data
        'feature_selection': ['chi2', 'none', 'random'], # Sets the feature selection strategy to be used
        'p_threshold': [0.01, 0.05, 0.1],               # P-value threshold for chi-squared test on bag-of-words
        'support_probabilities': [True, False],         # If True, trains a LogisticRegression to support predict_proba()
        'n_jobs': [n_jobs_model_val],                             # Number of jobs to run in parallel for fit and predict
        'random_state': [random_state_val],                       # Seed for random number generation
    }
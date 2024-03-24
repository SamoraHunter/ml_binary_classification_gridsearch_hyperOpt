class SummaryClassifier_class():
    algorithm_implementation = SummaryClassifier

    method_name = 'SummaryClassifier'

    parameter_space = {
        'summary_functions': ['mean', 'std', 'min', 'max', 'median', 'sum', 'skew', 'kurt', 'var', 'mad', 'sem', 'nunique', 'count'],
        'summary_quantiles': [None, [0.25, 0.5, 0.75]],  # Optional list of series quantiles to calculate
        'estimator': [None, RandomForestClassifier(n_estimators=200)],  # An sklearn estimator to be built using the transformed data
        'n_jobs': [n_jobs_model_val],  # Number of jobs to run in parallel for fit and predict
        'random_state': [random_state_val],  # Seed for random number generation
    }
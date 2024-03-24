class TemporalDictionaryEnsemble_class():
    algorithm_implementation = aeon.classification.dictionary_based._tde.TemporalDictionaryEnsemble
    

    method_name = 'TemporalDictionaryEnsemble'

    parameter_space = {
        'n_parameter_samples': [100, 250, 500],          # Number of parameter combinations to consider
        'max_ensemble_size': [25, 50, 100],              # Maximum number of estimators in the ensemble
        'max_win_len_prop': [0.5, 1.0],                 # Maximum window length as a proportion of series length
        'min_window': [5, 10, 15],                      # Minimum window length
        'randomly_selected_params': [25, 50, 75],       # Number of randomly selected parameters before GP parameter selection
        'bigrams': [True, False, None],                 # Whether to use bigrams
        'dim_threshold': [0.7, 0.85, 0.95],             # Dimension accuracy threshold for multivariate data
        'max_dims': [10, 20, 30],                       # Max number of dimensions per classifier for multivariate data
        'time_limit_in_minutes': time_limit_param,           # Time contract to limit build time in minutes
        'contract_max_n_parameter_samples': [100, 250, 500],  # Max number of parameter combinations to consider with time limit
        'typed_dict': [True, False],                    # Use a numba typed Dict to store word counts
        #'save_train_predictions': [True, False],        # Save the ensemble member train predictions in fit
        'n_jobs': [n_jobs_model_val],                             # Number of jobs for parallel processing
        'random_state': [random_state_val],                     # Random seed for random number generation
    }
class CNNClassifier_class():
    algorithm_implementation = CNNClassifier
    
    method_name = 'CNNClassifier'
    
    #nb revisit params dependent on width of the dataset
    
    parameter_space = {
        
        #'n_layers': [2, 3, 4],
        #'kernel_size': [3, 5, 7],
        #'n_filters': [[6, 12], [8, 16], [10, 20]],
        #'avg_pool_size': [2, 3, 4],
        'activation': ['sigmoid', 'relu'],
        'padding': ['valid'],
        #'strides': [1, 2],
        'dilation_rate': [1, 2],
        'use_bias': [True],
        'random_state': [random_state_val],
        'n_epochs': log_epoch,
        'batch_size': [16, 32, 64],
        'verbose': [ verbose_param],
        'loss': ['binary_crossentropy'],
        'metrics': ['accuracy'],
        #'save_best_model': [True, False],
        #'save_last_model': [True, False],
        #'best_file_name': ['best_model', 'top_model'],
        #'last_file_name': ['last_model', 'final_model'], 
    }
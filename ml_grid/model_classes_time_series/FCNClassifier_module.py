class FCNClassifier_class():
    algorithm_implementation = FCNClassifier
    
    method_name = 'FCNClassifier'
    
    parameter_space = {
        'n_layers': [3],
        'n_filters': [128, 256, 128],
        'kernel_size': [8, 5, 3],
        'dilation_rate': [1],
        'strides': [1],
        'padding': ['same'],
        'activation': ['relu'],
        'use_bias': [True],
        'n_epochs': log_epoch,
        'batch_size': [16],
        'use_mini_batch_size': [True],
        'random_state': [random_state_val],
        'verbose': [verbose_param],
        'loss': ['categorical_crossentropy'],
        'metrics': [None],
        'optimizer': [keras.optimizers.Adam(0.01), keras.optimizers.SGD(0.01)], 
        #'n_jobs':[1] #not a param
        #'file_path': ['./'],
        #'save_best_model': [False],
        #'save_last_model': [False],
        #'best_file_name': ['best_model'],
        #'last_file_name': ['last_model'],
        #'callbacks': [None]
    }
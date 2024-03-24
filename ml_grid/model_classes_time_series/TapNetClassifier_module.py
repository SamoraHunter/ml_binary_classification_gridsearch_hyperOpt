class TapNetClassifier_class():
    algorithm_implementation = TapNetClassifier

    method_name = 'TapNetClassifier'

    parameter_space = {
        'filter_sizes': [(256, 256, 128), (128, 128, 64)],  # Sets the kernel size argument for each convolutional block. Controls the number of convolutional filters and the number of neurons in attention dense layers.
        'kernel_size': [(8, 5, 3), (4, 3, 2)],  # Controls the size of the convolutional kernels.
        'layers': [(500, 300), (400, 200)],  # Size of dense layers.
        #'reduction': [16, 32],  # Divides the number of dense neurons in the first layer of the attention block.
        'n_epochs': log_epoch,  # Number of epochs to train the model.
        'batch_size': [16, 32],  # Number of samples per update.
        'dropout': [0.5, 0.3, 0.2],  # Dropout rate, in the range [0, 1).
        'dilation': [1, 2],  # Dilation value.
        'activation': ['sigmoid', 'relu'],  # Activation function for the last output layer.
        'loss': ['binary_crossentropy', 'categorical_crossentropy'],  # Loss function for the classifier.
        'optimizer': [keras.optimizers.Adam(0.01), keras.optimizers.SGD(0.01)],  # Gradient updating function for the classifier.
        'use_bias': [True, False],  # Whether to use bias in the output dense layer.
        'use_rp': [True, False],  # Whether to use random projections.
        'use_att': [True, False],  # Whether to use self-attention.
        'use_lstm': [True, False],  # Whether to use an LSTM layer.
        'use_cnn': [True, False],  # Whether to use a CNN layer.
        'verbose': [ verbose_param],  # Whether to output extra information.
        'random_state': [random_state_val],  # Seed for random.
    }
class SignatureClassifier_class():
    algorithm_implementation = SignatureClassifier

    method_name = 'SignatureClassifier'

    parameter_space = {
#         'estimator': [RandomForestClassifier(n_estimators=100), DecisionTreeClassifier()],
#         'augmentation_list': [('basepoint', 'addtime'), ('addtime',)],
#         'window_name': ['dyadic', 'sliding', 'expanding'],
#         'window_depth': [2, 3, 4],
#         'window_length': [None, 50, 100],
#         'window_step': [None, 1, 5],
#         'rescaling': [None, 'standard', 'min-max'],
#         'sig_tfm': ['signature', 'logsignature'],
#         'depth': [3, 4, 5],
        'random_state': [0],
    }
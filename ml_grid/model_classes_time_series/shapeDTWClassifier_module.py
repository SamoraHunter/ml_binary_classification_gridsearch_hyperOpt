
class ShapeDTW_class():
    
    algorithm_implementation = ShapeDTW
    
    method_name = 'ShapeDTW'
    
    parameter_space = {
        
        'n_neighbours': [-1],
        'subsequence_length': ['sqrt(n_timepoints)'],
        'shape_descriptor_function': ['raw'],
        'params': [None],
        'shape_descriptor_functions': [['raw', 'derivative']],
        'metric_params': [None],
        'n_jobs': [n_jobs_model_val]
        
    }
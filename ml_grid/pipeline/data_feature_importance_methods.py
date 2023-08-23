from ml_grid.pipeline.data_feature_methods import feature_methods

#rename this class

class feature_importance_methods():
    
    
    
    def __init__(self):
        
        
        """_summary_
        """
    
    
    def handle_feature_importance_methods(self, target_n_features, X_train, X_test, y_train, X_test_orig):
        
            
        #can implement further methods here on features
        
        features = feature_methods.getNfeaturesANOVAF(self, n = target_n_features, X_train = X_train, y_train = y_train)

        print(f"target_n_features: {target_n_features}")

        X_train = X_train[features]
        
        X_test = X_test[features]
        
        X_test_orig = X_test_orig[features]
        
        
        
        return X_train, X_test, X_test_orig
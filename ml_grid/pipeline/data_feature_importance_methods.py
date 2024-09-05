from ml_grid.pipeline.data_feature_methods import feature_methods

# rename this class


class feature_importance_methods:

    def __init__(self):
        """_summary_"""

    def handle_feature_importance_methods(
        self, target_n_features, X_train, X_test, y_train, X_test_orig, ml_grid_object
    ):
        """
        Handles feature importance methods on the data.

        Parameters
        ----------
        target_n_features : int
            Target number of features to reduce to.
        X_train : pd.DataFrame
            Pandas DataFrame of training data X.
        X_test : pd.DataFrame
            Pandas DataFrame of testing data X.
        y_train : pd.Series
            Pandas Series of training data y.
        X_test_orig : pd.DataFrame
            Pandas DataFrame of original testing data X.

        Returns
        -------
        X_train : pd.DataFrame
            Pandas DataFrame of training data X with reduced features.
        X_test : pd.DataFrame
            Pandas DataFrame of testing data X with reduced features.
        X_test_orig : pd.DataFrame
            Pandas DataFrame of original testing data X with reduced features.
        """

        feature_method = ml_grid_object.local_param_dict.get("feature_selection_method")

        if(feature_method == 'anova' or feature_method == None):
            
            print("feature_method ANOVA")

            features = feature_methods.getNfeaturesANOVAF(
                self, n=target_n_features, X_train=X_train, y_train=y_train
                )

        elif feature_method == 'markov_blanket':

            print("feature method Markov")
        
            features = feature_methods.getNFeaturesMarkovBlanket(
                
            self, n=target_n_features, X_train=X_train, y_train=y_train
            )

        print(f"target_n_features: {target_n_features}")

        X_train = X_train[features]

        X_test = X_test[features]

        X_test_orig = X_test_orig[features]

        return X_train, X_test, X_test_orig

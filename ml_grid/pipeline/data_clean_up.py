

import re

from ml_grid.util.global_params import global_parameters


class clean_up_class():
    
    def __init__(self):
        
        self.global_params = global_parameters()
        
        self.verbose = self.global_params.verbose
        
        self.rename_cols = self.global_params.rename_cols
        
        #print mass debug statement for cleaning procedures?
        
        #pass
    
    
    
    def handle_duplicated_columns(self, X):
        
        if(self.verbose > 1):
            print("dropping duplicated columns")
        
        X = X.loc[:,~X.columns.duplicated()].copy()
        
        return X
    
    
    def screen_non_float_types(self, X):
        
        if(self.verbose > 1):
            print("Screening for non float data types:")
            #types = []
            for col in X.columns:
                if(X[col].dtype != int and X[col].dtype != float):
                    print(col)
        

    def handle_column_names(self, X):
        
        """Rename columns to remove bad characters, xgb related"""
        if(self.rename_cols):
            regex = re.compile(r"\[|\]|<", re.IGNORECASE)
            X.columns = [regex.sub("_", col) if any(X in str(col) for X in set(('[', ']', '<'))) else col for col in X.columns.values]
            
            return X
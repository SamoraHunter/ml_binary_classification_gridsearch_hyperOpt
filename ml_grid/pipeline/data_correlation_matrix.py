import numpy as np
import pandas as pd


def handle_correlation_matrix(local_param_dict, drop_list, df):
    
        temp_col_list = list(df.select_dtypes(include=[float, int]).columns)

        corr_matrix = pd.DataFrame(np.corrcoef(df.select_dtypes(include=[float, int]).values, rowvar=False), columns=temp_col_list).abs()
        
        # Create a True/False mask and apply it
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        tri_df = corr_matrix.mask(mask)
        # List column names of highly correlated features (r > %user_defined% )
        corr_to_drop = [c for c in tri_df.columns if any(tri_df[c] > local_param_dict.get('corr'))]

       
        print(f"Identified {len(corr_to_drop)} correlated features to drop at >{local_param_dict.get('corr')}")
        drop_list.extend(corr_to_drop)
        
        
        return drop_list
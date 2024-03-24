def feature_encoding_to_feature_names(df, results_df):
    """Provide input df and results df"""
    
    # Extracting the list from the results dataframe
    f_list = eval(results_df.iloc[0]['f_list'])[0]
    
    # Extracting column names from the DataFrame
    column_names = df.columns.tolist()
    
    # Filtering strings based on the map
    filtered_strings = [column_name for column_name, value in zip(column_names, f_list) if value == 1]
    
    return filtered_strings


import pickle


def handle_percent_missing(local_param_dict, all_df_columns, drop_list):
    
    with open('percent_missing_dict.pickle', 'rb') as handle:
            percent_missing_dict = pickle.load(handle)


    percent_missing_threshold = local_param_dict.get('percent_missing')
    percent_missing_drop_list = []
    for col in all_df_columns:
        try:
            if(percent_missing_dict.get(col)>percent_missing_threshold):
                percent_missing_drop_list.append(col)
        except:
            pass
            
    print(f"Identified {len(percent_missing_drop_list)} at > {percent_missing_threshold} threshold")
    drop_list.extend(percent_missing_drop_list)
        
        
    return drop_list



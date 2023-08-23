

import logging
import pathlib




class log_folder():


        def __init__(self, local_param_dict, additional_naming, base_project_dir):
            
            str_b = ''
            for key in local_param_dict.keys():
                if(key != 'data'):
                    str_b = str_b + '_' + str(local_param_dict.get(key))
                else:
                    for key in local_param_dict.get('data'):
                        str_b = str_b + str(int(local_param_dict.get('data').get(key)))
            
            global_param_str = str_b
            #global_param_str = str(global_param_dict).replace("{", "").replace("}", "").replace(":", "").replace(" ", "").replace(",", "").replace("'", "_").replace("__", "_").replace("'","").replace(",","").replace(": ", "_").replace("{","").replace("}","").replace("True","T").replace("False", "F").replace(" ","_").replace("[", "").replace("]", "").replace("_","")
            
            print(global_param_str)




            
            
            
            log_folder_path = f"{global_param_str + additional_naming}/logs/"
            
            pathlib.Path(base_project_dir+log_folder_path).mkdir(parents=True, exist_ok=True) 
            
            full_log_path = f"{base_project_dir+global_param_str + additional_naming}/logs/log.log"
            
            logging.basicConfig(filename=full_log_path)
            stderrLogger=logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)
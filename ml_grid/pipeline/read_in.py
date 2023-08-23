
import pandas as pd


class read():
    
    def __init__(self, input_filename):
        
        filename = input_filename

        print(f"Init main on {filename}")
        
        self.raw_input_data = pd.read_csv(filename)
        
        
        
        
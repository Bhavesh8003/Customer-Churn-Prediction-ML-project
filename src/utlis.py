'''This file will have all the common things that we 
are going to probably trying to import or use.
the common functionalities which the entire project can use 
'''
import os
import sys
import dill #help to create pickle file
from src.exception import CustomException
import numpy as np
import pandas as pd

def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

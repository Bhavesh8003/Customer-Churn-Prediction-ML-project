'''
In this file we are handling the categorical values numerical values 
and creating entire pickle file
'''

import sys
import os
from dataclasses import dataclass
import numpy as np  
import pandas as pd
from sklearn.compose import  ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utlis import save_object  #just used for saving the pickle file

from src.exception import CustomException
from src.logger import logging



@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    #create all the pickle files which will be responsible for converting categorical into numerical, perform scaler and all.
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation based on different types of data
        '''
        
        try:
            
            numerical_columns = ['Count', 'Zip Code', 'Latitude', 'Longitude', 'Tenure Months','Monthly Charges', 'Churn Score', 'CLTV','Total Charges']
            categorical_columns = ['Gender','Senior Citizen', 'Partner', 'Dependents', 'Phone Service','Multiple Lines', 'Internet Service', 'Online Security','Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV','Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logging.info(f"Numerical columns:{numerical_columns}")
            logging.info(f"Categorical columns:{categorical_columns}")
            
            preprocessor= ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)  
                ]
                
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")
            
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = "Churn Value"
            numerical_columns = ['Count', 'Zip Code', 'Latitude', 'Longitude', 'Tenure Months','Monthly Charges', 'Churn Value', 'Churn Score', 'CLTV']
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )
               
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)

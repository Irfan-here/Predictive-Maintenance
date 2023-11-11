import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.utils import save_object


from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numeric_features = ['Air temperature [K]', 'Process temperature [K]', 
                                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
            categorical_features = ['Type', 'Failure Type']

            num_pipe = Pipeline(steps=[
                ("scaler", MinMaxScaler())
            ])

            cat_pipe = Pipeline(steps=[
                ('onehot', OneHotEncoder())
            ])

            logging.info('Numerical and Categorical Features Transformed')

            preprocessor = ColumnTransformer([
                ("num_pipe",num_pipe,numeric_features),
                ("cat_pipe",cat_pipe,categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) 
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Training and Testing Data")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_transformer_object()

            target_column_name = 'Target'
            numeric_features = ['Air temperature [K]', 'Process temperature [K]', 
                                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing Object on Train & Test DFs")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved Preprocessing Object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except:
            pass


        
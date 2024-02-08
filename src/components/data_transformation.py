import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        ''' 
        This is function is responsible for data transformation 

        '''
        
        logging.info("Entered into the get_data_transormer_object method")

        try:
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            ### To handle the missing values and performing standard scaler, and one hot encoding for categorical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), ## Imputer is to handle the missing values
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            ### Combining two pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessing object")

            ###-----------  Feature scaling starts here  -------------------------

            preprocessing_obj = self.get_data_transformer_object()

            ## Independent and dependent feature division and dropping columns
            target_column_name = ["math score"]
            numerical_columns = ['reading score', 'writing score']

            input_feature_train_df = train_df.drop(columns=target_column_name, axis=1)   ##--> X_train
            target_feature_train_df = train_df[target_column_name]  ##--> y_train

            input_feature_test_df  = test_df.drop(columns=target_column_name, axis=1)  ##--> X_test
            target_feature_test_df = test_df[target_column_name]  ##--> y_test

            logging.info("Applying preprocessing object in training and test datasets")

            ###------------ fit_transform and transform(scaling) --------------
            # Transformatting using preprocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatinating two arrays(input_feature_train_arr & y_train ; input_feature_test_arr & y_test) through numpy concatinate funtionality as if it will be required in future then it is fast to access in the array format
            train_arr = np.c_(input_feature_train_arr, np.array(target_feature_train_df))
            test_arr = np.c_(input_feature_test_arr, np.array(target_feature_test_df))

            logging.info("Saved preprocessing object")

            ### For saving the preprocessor pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
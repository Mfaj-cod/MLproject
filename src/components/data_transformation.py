import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.datatransformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_column = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # pipeline for numerical data transformation
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('numerical columns encoding completed')

            # pipeline for categorical data transformation
            categ_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('categorical columns encoding completed')
            
            # pipeline for transforming both data
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_column),
                ('categ_pipeline', categ_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train test data completed.')\
                
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            
            # dropping the target features because we don't do any kind of transformation to the target column
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)

            # target features
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            logging.info('applying preprocessor obj')

            # transforming the input features
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # np.c_[] concatenates the arrays column-wise ****
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # saving the preprocessor.pkl
            save_object(
                file_path=self.datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        

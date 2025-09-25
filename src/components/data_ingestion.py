import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# defining file paths
@dataclass
class dataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')


# main data injector class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig() # getting paths

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('read the dataset as df')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # Making dir

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # saving the df to the dir
            logging.info('Train test split initiated')

            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42) # splitting the data

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('ingestion of the data is completed.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj=DataIngestion() # creating obj with main data injector
    train_data, test_data = obj.initiate_data_ingestion() # capturing the returned values from initiate method

    data_transformation = DataTransformation() # with datatransformer
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation( #capturing the returned values
        train_data,
        test_data
    )

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer( #printing the best model name and the r2 score
        train_array=train_arr,
        test_array=test_arr
    ))

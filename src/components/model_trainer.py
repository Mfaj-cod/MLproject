import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('splitting ytraing and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],  # all rows, every column except the last column → input features
                train_array[:,-1],   # all rows, only the last column → target/labels
                test_array[:,:-1],   # same for test set → input features
                test_array[:,-1]     # same for test set → target/labels
            )


            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision tree": DecisionTreeRegressor(),
                "Gradient boosting": GradientBoostingRegressor(),
                "Linear regression": LinearRegression(),
                "K-nearest neighbours": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }


            params = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },

                "Decision tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },

                "Gradient boosting": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.8, 1.0],
                    'min_samples_split': [2, 5, 10]
                },

                "Linear regression": {
                    # Mostly parameter-free, but these are optional
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                },

                "K-nearest neighbours": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },

                "XGBoost": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'reg_alpha': [0, 0.01, 0.1],
                    'reg_lambda': [1, 1.5, 2.0]
                },

                "CatBoost": {
                    'iterations': [200, 500],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'l2_leaf_reg': [1, 3, 5, 7, 9]
                    # Note: CatBoost handles categorical automatically
                },

                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'loss': ['linear', 'square', 'exponential']
                }
            }




            # calling the evaluate function from utils.py
            model_report:dict=evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
                )
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException('No model is good enough')

            logging.info(f'Best model found on both training and testing dataset: {best_model}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_pred=predicted, y_true=y_test)

            return best_model_name, r2


        except Exception as e:
            raise CustomException(e, sys)

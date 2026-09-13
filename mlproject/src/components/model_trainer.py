import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    # save the model at this location
    trained_model_file_path:str=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info('split training a testing data')

            X_train, X_test, y_train, y_test = (
                # all rows and cols except last col
                train_array[:, :-1],
                # all rows with only the last col
                test_array[:, :-1],

                train_array[:, -1],
                test_array[:, -1],       
            )

            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },

                "Random Forest Regressor": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },

                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                },

                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },

                "XGBRegressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },

                "CatBoosting Regressor": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200],
                    'l2_leaf_reg': [3, 5]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.5],
                    'n_estimators': [50, 100, 200],
                    'loss': ['linear', 'square']
                }
            }

            model_report:dict=evaluate_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models,
                param=params
            )

            # best model score
            best_model_score = max(sorted(model_report.values()))
            # best name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('There is no model that is best')
            logging.info(f'Best Model on both train and test data')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_value = r2_score(y_test, predicted)
            return r2_value
        
        except Exception as e:
            raise CustomException(e, sys)
            
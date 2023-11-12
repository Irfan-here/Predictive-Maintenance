import sys
from dataclasses import dataclass
import os
from matplotlib.pyplot import cla

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Training and Testing Data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                      "Logistic Regression" : LogisticRegression(class_weight='balanced'),
                      "Random Forest": RandomForestClassifier(),
                      "Gradient Boost": GradientBoostingClassifier()
                      
                      }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                               models = models)
            

            
            
            
            # scores_list= model_report.values()
            # scores = []
            # best_model = None

            # for i in scores_list:
            #     scores.append(i[0])

            # for j in model_report:
            #     if max(scores) in model_report[j]:
            #         best_model = j
            #         break

            

            scores_list = model_report.values()
            scores = [i[0] for i in scores_list]

            best_model = max(model_report, key=lambda k: model_report[k][0])

            best_Model = models[best_model]

            if max(scores) < 0.6:
                raise CustomException("No Model is performing well")
            
            logging.info("Best Model Found on Train and Test")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_Model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            f1 = f1_score(y_test, predicted)
            confusion = confusion_matrix(y_test, predicted)

            scores = [accuracy,precision,recall,f1,confusion]

            return (scores, best_model)

        except Exception as e:
            raise CustomException(e,sys)
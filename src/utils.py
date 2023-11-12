import os, sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)

            scores = [accuracy,precision,recall,f1,confusion]

            report[list(models.keys())[i]] = scores

        return report

            

    except Exception as e:
        raise CustomException(e,sys)
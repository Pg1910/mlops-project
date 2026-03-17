import os
import numpy as np
import pandas as pd
import pickle
import logging
import json 
from sklearn.metrics import accuracy_score , precision_score , recall_score ,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import yaml
from dvclive import Live



log_dir = 'logs'
os.makedirs(log_dir , exist_ok=True)

#logging configuration 
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir , 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path : str) -> dict:
    """load params from a YAML file."""
    try:
        with open(params_path , 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('parametres retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('file not fund  %s ', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error %s ',e)
        raise
    except Exception as e:
        logger.error('unexpected error %s',e)
        raise


def load_model(file_path : str):
    """load the model from the model dir"""
    try:
        with open(file_path , 'rb') as file:
            model = pickle.load(file)
        logger.debug('modle found in the dir %s' ,file_path)
        return model
    except FileNotFoundError :
        logger.error('file not found %s ', file_path)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the model : %s',e)
        raise


def load_data(file_path : str) -> pd.DataFrame:
    """load data from a csv"""
    try :
        df = pd.read_csv(file_path)
        logger.debug('data loaded from %s with shape %s' , file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file : %s',e)
        raise
    except FileNotFoundError as e:
        logger.error('failed not found : %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the data : %s',e )
        raise

def evaluate_model (clf , X_test :np.ndarray , y_test :np.ndarray) -> dict:
    '''evaluate the model and return the evaluation metrics '''
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[: , 1]
        accuracy = accuracy_score(y_test , y_pred)
        precision = precision_score(y_test , y_pred)
        recall = recall_score(y_test ,y_pred)
        auc = roc_auc_score(y_test , y_pred_proba)
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall' : recall,
            'auc' : auc
        }
        logger.debug ('model evaluation metrics calcualted ')
        return metrics_dict
    except Exception as e:
        logger.error('error during the model evaluation %s' ,e)
        raise

def save_metrics(metrics:dict , file_path :str) -> None:
    """ save the evalution metrics to a json file"""
    try:
        #ensure the dir exists
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        with open (file_path , 'w') as file:
            json.dump(metrics , file ,indent = 4)
        logger.debug('metrics saved to %s ', file_path)
    except Exception as e:
        logger.error('error occured while saving the metrics : %s ', e)
        raise

def main ():
    try:
        params = load_params(params_path= 'params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./proj2/data/processed/test_tfidf.csv')
        X_test = test_data.iloc[: , :-1].values
        y_test = test_data.iloc[: , -1].values
        metrics = evaluate_model(clf , X_test , y_test)

        # experimnet tracking using dvc live
        with Live(save_dvc_exp = True) as dvc_live: #type: ignore
                dvc_live.log_metric('accuracy' , metrics['accuracy'])
                dvc_live.log_metric('precision', metrics['precision'])
                dvc_live.log_metric('recall'   , metrics['recall'])
                dvc_live.log_params(params)

        save_metrics(metrics , './proj2/reports/metrics.json')
    except Exception as e:
        logger.error('failed to complete the model  evaluation process : %s ',e)
        print(f"Error , {e}")
if __name__ == '__main__':
    main()

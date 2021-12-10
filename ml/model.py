import warnings
import joblib
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import datetime
from datetime import date
from joblib import dump,load
from flask import Flask
from array import array


app = Flask(__name__)


class DataHandler:
    '''
        Get data from sources
    '''
    def __init__(self):
        self.csvfile1 = None
        self.csvfile2 = None
        self.gouped_data = None


class FeatureRecipe:
    
    '''Feature processing class'''
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.continuous = None
        self.categorical = None
        self.discrete = None
        self.datetime = None


class FeatureExtractor:
    
    '''Feature Extractor class'''
    
    def __init__(self, data: pd.DataFrame, flist: list):
        '''
            Input : pandas.DataFrame, feature list to drop
            Output : X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        '''
        data.drop(array(flist), axis=1, inplace=True)


class ModelBuilder:
    '''
        Class for train and print results of ml model
    '''
    model = None
    model_path = "decision_rf.joblib"

    def __init__(self, model_path: str = None, save: bool = None):
        pass
    def __repr__(self):
        pass
    def train(self, X, Y):
        pass
    def predict_test(self, X) -> np.ndarray:
        pass
    def predict_from_dump(self, X) -> np.ndarray:
        pass
    def save_model(self, path:str):
        joblib.dump(self.model,path)
    def print_accuracy(self):
        self.model.predict()
    def load_model(self):
        try:
            #load model
            model = joblib.load(self.model_path)
        except:
            raise Exception("No model found")        


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
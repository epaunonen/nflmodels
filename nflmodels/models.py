import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss, f1_score, RocCurveDisplay
import xgboost as xgb

class NFLModel():
    
    def __init__(self):
        self.classifier = None
    
    def fit(self, X_train : pd.DataFrame, y_train : pd.DataFrame):
        self.classifier.fit(X_train, y_train)
        
    def validate(self, X_val : pd.DataFrame, y_val : pd.DataFrame):
        y_pred = self.classifier.predict(X_val)
        y_pred_proba = self.classifier.predict_proba(X_val)
        return y_pred, y_pred_proba
        
    def fit_validate(self, X_train : pd.DataFrame, y_train : pd.DataFrame, X_val : pd.DataFrame, y_val : pd.DataFrame):
        self.fit(X_train, y_train)
        self.validate(X_val, y_val)
        
    
    
    
class NextPlayModel(NFLModel):
        
    def __fit(self, X_train : pd.DataFrame, y_train : pd.DataFrame):
        return


class FieldGoalModel(NFLModel):
    
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000)
        
    def validate(self, X_val: pd.DataFrame, y_val: pd.DataFrame):
        y_pred, y_pred_proba = super().validate(X_val, y_val)
        
        print('On validation set')
        print('ROC AUC: {:.3f}\nMSE: {:.3f}\nLog loss: {:.3f}'.format(roc_auc_score(y_val, y_pred_proba[:,1]), 
                                                                                 mean_squared_error(y_val, y_pred), 
                                                                                 log_loss(y_val, y_pred_proba)))
        
        return y_pred, y_pred_proba


class EPModel(NFLModel):
        
    def __fit(self, X_train : pd.DataFrame, y_train : pd.DataFrame):
        return
    
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss, f1_score
import xgboost as xgb

def create_field_goal_predictor(X_train : pd.DataFrame, y_train : pd.DataFrame, X_val : pd.DataFrame = None, y_val : pd.DataFrame = None, print_eval_metrics = False):
    '''
    
    '''
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        y_pred = clf.predict(X_val)
        y_pred_proba = clf.predict_proba(X_val)
        
        if print_eval_metrics:
            print('On validation set')
            print('ROC AUC: {:.3f}\nMSE: {:.3f}\nLog loss: {:.3f}'.format(roc_auc_score(y_val, y_pred_proba[:,1]), 
                                                                                 mean_squared_error(y_val, y_pred), 
                                                                                 log_loss(y_val, y_pred_proba)))
        return clf, y_pred, y_pred_proba
    else:
        return clf
    

def create_next_play_predictor(X_train : pd.DataFrame, y_train : pd.DataFrame, X_val : pd.DataFrame = None, y_val : pd.DataFrame = None, print_eval_metrics = False):
    '''
    
    '''
    
    clf = xgb.XGBClassifier(n_jobs=-1, n_estimators=100, learning_rate=0.1, eval_metric='mlogloss', min_child_weight=2)
    clf.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        y_pred = clf.predict(X_val)
        y_pred_proba = clf.predict_proba(X_val)
        
        if print_eval_metrics:
            print('On validation set')
            print('ROC AUC: {:.3f}\nF1 score: {:.3f}\nMSE: {:.3f}\nLog loss: {:.3f}'.format(
                roc_auc_score(pd.get_dummies(y_val, columns=y_val.columns.values), y_pred_proba, average='weighted', multi_class='ovo'), 
                f1_score(y_val, y_pred, average='weighted'),
                mean_squared_error(y_val, y_pred), 
                log_loss(y_val, y_pred_proba))
            )
            
        return clf, y_pred, y_pred_proba
    else:
        return clf
    

def create_ep_predictor(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame = None, y_val: pd.DataFrame = None, print_eval_metrics = False):
    '''
    
    '''
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    
    return
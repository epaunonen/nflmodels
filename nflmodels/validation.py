import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.metrics import RocCurveDisplay

def valplot_fieldgoal(X_val : pd.DataFrame, y_val : pd.DataFrame, y_pred, y_pred_proba):
    fig, ax = plt.subplots(1, 3, figsize=(21,6))

    # 1. ROC curve
    RocCurveDisplay.from_predictions(y_val, y_pred_proba[:,1], ax=ax[0], color='maroon')
    ax[0].set_title('ROC Curve, validation set')
    ax[0].plot([0,1], [0,1], color='black', linestyle='--', linewidth=2)

    # 2. Accuracy of predicted probabilities
    # For predicted success %, what portion of attempts were actually successful
    # Yellow line decipts optimal performance where the predicted success chance matches the proportion of successful attempts 
    sns.histplot(x=y_pred_proba[:,1], hue=y_val, ax=ax[1], multiple='fill', stat='probability', palette='RdBu', hue_norm=(0,1), bins=12, alpha=0.9)
    ax[1].plot([0,1], [0,1], color='yellow', linestyle='--', linewidth=2)
    ax[1].set_xlim(0.3, 1)
    ax[1].set_ylim(0.3, 1)
    ax[1].set_xlabel('Predicted success chance')
    ax[1].set_ylabel('Distribution of true results')
    ax[1].set_title('Predicted % accuracy, validation set')

    # 3. Predicted success by distance to end zone, validation set
    sns.lineplot(data=X_val, x='yardline_100', y=y_pred_proba[:,1], ax=ax[2], palette='RdBu', hue_norm=(0,1), #legend=None,
                #hue=X_val[[item for item in list(clf.feature_names_in_) if item.startswith('kicker_player_name_')]].idxmax(axis=1).str.split('_').str[-1]
                hue='closed'
                )
    ax[2].set_ylabel('Success chance')
    ax[2].set_xlabel('Yards to goal line')
    ax[2].set_ylim(0,1)
    ax[2].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax[2].set_xlim(0, 60)
    ax[2].set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax[2].set_title('Predicted success chance by distance to end zone, validation set')

    plt.tight_layout()
    plt.show()

def valplot_nextplay(X_val : pd.DataFrame, y_val : pd.DataFrame, y_pred, y_pred_proba):
    return

def valplot_ep(X_val : pd.DataFrame, y_val : pd.DataFrame, y_pred, y_pred_proba):
    return
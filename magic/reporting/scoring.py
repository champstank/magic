import sys
sys.path.insert(0, "/data/magic/")
import numpy as np
from data.review import is_regression
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score, f1_score, r2_score
from sklearn.metrics import precision_score, recall_score, mean_absolute_error, mean_squared_error, median_absolute_error
from scipy.stats import pearsonr

def scoring(y,ypred_class,ypred_prob): 
    """
    This function will create a scored dictionary of your predictions
    Params:
        y np.array actual value
        ypred_class np.array predicted values
        ypred_prob np.array predicted probabilities of values
    Returns:
        results dict dictionary of all results
    """

    #accuracy_score(y,ypred_class)
    results={}
    if is_regression(y):
        print('Regression detected!')
        results['neg_mean_absolute_error'] = mean_absolute_error(y,ypred_class)
        results['neg_median_absolute_error'] =  median_absolute_error(y,ypred_class)
        results['neg_squared_absolute_error'] = mean_squared_error(y,ypred_class)
        results['r2'] = r2_score(y,ypred_class)
        results['pearson-r'] = pearsonr(y,ypred_class)[0]
        results['pearson-r-pval'] = pearsonr(y,ypred_class)[1]
    else:   # must be binary
        if len(np.unique(y))<2:
            print('Binary detected!')
            results['auc_score'] = roc_auc_score(y,ypred_prob)
            results['pearson-r'] = pearsonr(y,ypred_prob)[0]
            results['pearson-r-pval'] = pearsonr(y,ypred_prob)[1]
            results['average_precision_score'] = average_precision_score(y,ypred_class)
            results['f1_score'] = f1_score(y,ypred_class)
            results['precision_score'] = precision_score(y,ypred_class)
            results['recall_score'] = recall_score(y,ypred_class)
        else:
            print('Multi-class detected!')

        results['accuracy'] = accuracy_score(y,ypred_class)
        results['confusion_matrix'] = confusion_matrix(y,ypred_class)
        results['observation_count'] = len(y)
        results['label_balance'] = np.mean(y)
    return results

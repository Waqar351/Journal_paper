import numpy as np
import cvxpy as cvx
import pandas as pd

def FM(train_scores, test_scores, train_labels, nclasses):    
    CM = np.zeros((nclasses, nclasses))
    y_cts = np.array(pd.DataFrame(train_labels).value_counts())
    p_yt = y_cts / train_labels.shape[0]
    
    for i in range(nclasses):        
        idx = np.where(train_labels == i)[0]
        CM[:, i] += np.sum(train_scores[idx] > p_yt, axis=0) 
    CM = CM / y_cts
    p_y_hat = np.sum(test_scores > p_yt, axis = 0) / test_scores.shape[0]
    return p_y_hat[0]
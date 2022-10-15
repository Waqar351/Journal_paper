import numpy as np
import pandas as pd
from getTPRFPR import getTPRFPR

"""Median Sweep

It quantifies events based on their scores, applying Median Sweep (MS) method, according to Forman (2006).
 
Parameters
----------
test : array
    A numeric vector of scores predicted from the test set.
TprFpr : matrix
    A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
    
Returns
-------
array
    the class distribution of the test. 
 """
#def MS_method(test_scores, tprfpr):
def MS_method(test_scores, tprfpr):

    '''
    #unique_scores = np.unique(test_scores['scores'])

    unique_scores  = np.arange(0.01,1,0.01)  #threshold values from 0.01 to 0.99  
    prevalances_array = []
    
    for threshold in unique_scores:
        
        
        threshold =  round(threshold,2)
        
        record = tprfpr[tprfpr['threshold'] == threshold]
        
        #tpr = float(record['tpr'])
        tpr = record['tpr']
        
        #fpr = float(record['fpr'])
        fpr = record['fpr']
    
        batch_size = len(test_scores) 
    
        test = test_scores.astype(float)
    
        class_prop = len(np.where(test >= threshold)[0])/batch_size
           
        

        if (tpr - fpr) == 0:
            #pos_prop = class_prop
            prevalances_array.append(class_prop)
            

        else:

            #pos_prop = round((class_prop - fpr)/(tpr - fpr),2)  
    
            final_prevalence = round((class_prop - fpr)/(tpr - fpr),2)  
        
            prevalances_array.append(final_prevalence)  
            
    
    pos_prop = np.median(prevalances_array)

    pos_prop = round(pos_prop,2)
    
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop
    
  
    
    return pos_prop'''
    
    
    unique_scores  = np.arange(0.01,1,0.01)  #threshold values from 0.01 to 0.99  
    prevalances_array = []
    
    for i in unique_scores:
        
        
        threshold =  round(i,2)
        
        record = tprfpr[tprfpr['threshold'] == threshold]
        
        tpr = float(record['tpr'])
        
        fpr = float(record['fpr'])
    
        batch_size = len(test_scores) 
    
        test = test_scores.astype(float)
    
        class_prop = len(np.where(test >= threshold)[0])/batch_size
           
        

        if (tpr - fpr) == 0:
            #pos_prop = class_prop
            prevalances_array.append(class_prop)
            

        else:

            #pos_prop = round((class_prop - fpr)/(tpr - fpr),2)  
    
            final_prevalence = round((class_prop - fpr)/(tpr - fpr),2)  
        
            prevalances_array.append(final_prevalence)  
            
    
    pos_prop = np.median(prevalances_array)

    pos_prop = round(pos_prop,2)
    
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop
    
  
    
    return pos_prop
    

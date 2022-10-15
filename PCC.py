import numpy as np

def PCC(calib_clf,test_data, thr = 0.5):
    """Probabilistic Classify & Count (PCC)

    It quantifies events based on Calibrated classifier and correct the estimate using TPR and FPR, applying Probabilistic Classify & Count (PCC) method, according to Bella (2010).
    
    Parameters
    ----------
    calib_clf : Object
        Calibrated classifier previously trained from some training set partition.
    Test_data : Dataframe 
        A DataFrame of the test data. 
    thr : float  
        The threshold value for hard predictions. Default value = 0.5.
    
    Returns
    -------
    array
        the class distribution of the test. 
    """

    calibrated_predictions = calib_clf.predict_proba(test_data)[:,1]
    
    pcc_count = np.sum(calibrated_predictions[calibrated_predictions > thr])
    pos_prop = np.round(pcc_count/len(calibrated_predictions),2)

    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop
    
    return  pos_prop

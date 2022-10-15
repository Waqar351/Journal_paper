import numpy as np

def classify_count(test_scores,thr=0.5):
    """Classify & Count (CC)

    It quantifies events based on trained classifier, applying Classify & Count (ACC) method, according to Forman (2005).
    
    Parameters
    --------
    Test scores : array
        A numeric vector of scores predicted from the test set. 
    thr : float  
        The threshold value for hard predictions. Default value = 0.5.
    
    Returns
    -------
    array
        the class distribution of the test. 
    """
    
    count = len(np.where(test_scores >= thr)[0])      #Faster than using for loop below
    #count = len([i for i in test_scores if i >= thr])
    
    pos_prop = np.round(count/len(test_scores),2)
    
    return pos_prop

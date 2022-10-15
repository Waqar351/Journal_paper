import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier



def scorer(dt, label, folds):
    """
    This function estiamtes the scores of the provided training set using k-fold stratified cross-validation
    
    Parameters
    -----------
    dt :  dataframe
        A dataframe of training data.
    lable : vector
        It contains the label information of trainig data
    fold : integer
        number of folds 

    Returns
    ---------
    Dataframe
        estimated scores.
    """
    
    skf = StratifiedKFold(n_splits=folds)    
    clf=RandomForestClassifier(n_estimators=200)
    results = []
    class_labl = []
    
    for fold_i, (train_index,valid_index) in enumerate(skf.split(dt,label)):
        print('  Fold #%d' % (fold_i + 1))
        print('Training_len', len(dt))
        
        tr_data = pd.DataFrame(dt.iloc[train_index])   #Train data and labels
        tr_lbl = label.iloc[train_index]
        
        valid_data = pd.DataFrame(dt.iloc[valid_index])  #Validation data and labels
        valid_lbl = label.iloc[valid_index]
        
        clf.fit(tr_data, tr_lbl)
        
        results.extend(clf.predict_proba(valid_data)[:,1])     #evaluating scores
        class_labl.extend(valid_lbl)
        
        print('SCORES_Length:',len(results))
    
    scr = pd.DataFrame(results,columns=["scores"])
    scr_labl = pd.DataFrame(class_labl, columns= ["class"])
    scores = pd.concat([scr,scr_labl], axis = 1, ignore_index= False)
    
    return scores      

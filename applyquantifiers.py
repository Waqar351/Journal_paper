from CC import classify_count
from ACC import ACC
from PCC import PCC
from PACC import PACC
from HDy import Hdy
from X import X
from MAX import Max
from SMM import SMM  
from dys_method import dys_method
from sord import SORD_method
from MS import MS_method
from MS_2 import MS_method2
from T50 import T50
from EMQ import EMQ
from PWK import PWK
from GAC import GAC
from GPAC import GPAC
from FM import FM
from emq_quapy import EMQ_quapy
from schumar_model_fit import predict_quantifier_schumacher_github
import numpy as np
import pandas as pd
import time

def apply_quantifier(qntMethod, 
                    scores, 
                    p_score, 
                    n_score, 
                    train_labels, 
                    test_score, 
                    TprFpr, 
                    thr, 
                    measure, 
                    calib_clf, 
                    te_data, 
                    pwk_clf, 
                    schumacher_qnt, 
                    test_quapy, 
                    model_quapy):
    """This function is an interface for running different quantification methods.
 
    Parameters
    ----------
    qntMethod : string
        Quantification method name
    p_score : array
        A numeric vector of positive scores estimated either from a validation set or from a cross-validation method.
    n_score : array
        A numeric vector of negative scores estimated either from a validation set or from a cross-validation method.
    test : array
        A numeric vector of scores predicted from the test set.
    TprFpr : matrix
        A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
    thr : float
        The threshold value for classifying and counting. Default is 0.5.
    measure : string
        Dissimilarity function name used by the DyS method. Default is "topsoe".
    calib_clf : object
        A calibrated classifier used when PCC or PACC methods are called by the main experimental setup.
    te_data : dataframe
        A dataframe of test data.
    pwk_clf : object
        Nearest Neighbour classifier used only when PWK method is called. 
    schumacher_qnt : class
        contains functions of quantification methods used in schumacher paper
    test_quapy : dataframe
        A dataframe of test sample for Quapy package.
    model_quapy : object
        EMQ Model fitted using Quapy package (base RF calssifier)
    Returns
    -------
    array
        the class distribution of the test calculated according to the qntMethod quantifier. 
    """
    
    schumi_quantifiers = ['readme', 'HDx', 'FormanMM', 'CDE', 'EM']#, 'FM' #, 'GPAC', 'GAC'] quantifiers from schumacher paper

    if qntMethod in schumi_quantifiers:
        return predict_quantifier_schumacher_github(schumacher_qnt, te_data)
    if qntMethod == "cc":
        return classify_count(test_score, thr)
    if qntMethod == "acc":        
        return ACC(test_score, TprFpr)
    if qntMethod == "emq":          # This EMQ method is used from Quapy packages
        return EMQ_quapy(test_quapy, model_quapy)
    if qntMethod == "smm":
        return SMM(p_score, n_score, test_score)
    if qntMethod == "hdy":
        return Hdy(p_score, n_score, test_score)
    if qntMethod == "dys_Ts":
        return dys_method(p_score, n_score, test_score,measure)
    if qntMethod == "sord":
        return SORD_method(p_score, n_score,test_score)
    if qntMethod == "ms":
        return MS_method(test_score, TprFpr)
    if qntMethod == "ms2":
        return MS_method2(test_score, TprFpr)
    if qntMethod == "max":
        return Max(test_score, TprFpr)
    if qntMethod == "x":
        return X(test_score, TprFpr)
    if qntMethod == "t50":
        return T50(test_score, TprFpr)
    if qntMethod == "pcc":
        return PCC(calib_clf, te_data,thr)
    if qntMethod == "pacc":
        return PACC(calib_clf, te_data, TprFpr, thr)
    if qntMethod == "PWK":
        return PWK(te_data, pwk_clf)
    if qntMethod == "GAC":
        sc_p = np.append(np.array(p_score), n_score)
        sc_n = 1-sc_p
        scores = np.array(pd.concat([pd.DataFrame(sc_p), pd.DataFrame(sc_n)], axis=1))    
        sc_te = np.array(pd.concat([pd.DataFrame(test_score), pd.DataFrame(1-test_score)], axis=1))
        l_p = np.zeros(len(p_score))
        l_p[:] = 1
        l_n = np.zeros(len(n_score))
        return GAC(scores, sc_te, np.append(np.int0(l_p), np.int0(l_n)), 2)
    if qntMethod == "GPAC":
        sc_p = np.append(np.array(p_score), n_score)
        sc_n = 1-sc_p
        scores = np.array(pd.concat([pd.DataFrame(sc_p), pd.DataFrame(sc_n)], axis=1))
        l_p = np.zeros(len(p_score))
        l_p[:] = 1
        l_n = np.zeros(len(n_score))

        start = time.time()    
        te_scores = calib_clf.predict_proba(te_data)[:,1]  #estimating test sample scores
        sc_te = np.array(pd.concat([pd.DataFrame(te_scores), pd.DataFrame(1-te_scores)], axis=1))

        prop = GPAC(scores, sc_te, np.append(np.int0(l_p), np.int0(l_n)), 2)
        stop = time.time()
        return stop - start
        return prop
        #return GPAC(scores, sc_te, np.append(np.int0(l_p), np.int0(l_n)), 2)
        
    if qntMethod == "FM":
        sc_p = np.append(np.array(p_score), n_score)
        sc_n = 1-sc_p
        scores = np.array(pd.concat([pd.DataFrame(sc_p), pd.DataFrame(sc_n)], axis=1))    
        l_p = np.zeros(len(p_score))
        l_p[:] = 1
        l_n = np.zeros(len(n_score))

        start = time.time() 
        te_scores = calib_clf.predict_proba(te_data)[:,1]  #estimating test sample scores
        sc_te = np.array(pd.concat([pd.DataFrame(te_scores), pd.DataFrame(1-te_scores)], axis=1))
        
        prop = FM(scores, sc_te, np.append(np.int0(l_p), np.int0(l_n)), 2)
        
        stop = time.time()
        return stop - start
        return prop

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
from schumar_model_fit import predict_quantifier_schumacher_github
from temporary import temporary

def apply_quantifier(qntMethod, scores, p_score, n_score, train_labels, test_score, TprFpr, thr, measure, calib_clf, te_data, pwk_clf, schumacher_qnt):
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
    Returns
    -------
    array
        the class distribution of the test calculated according to the qntMethod quantifier. 
    """
    #schumi_quantifiers = ['readme', 'HDx', 'FormanMM', 'FM', 'CDE', 'GPAC', 'GAC'] #quantifiers from schumacher paper
    schumi_quantifiers = ['readme', 'HDx', 'FormanMM', 'CDE', 'EM', 'FM'] #quantifiers from schumacher paper

    if qntMethod in schumi_quantifiers:
        return predict_quantifier_schumacher_github(schumacher_qnt, te_data)[1]
    if qntMethod == "cc":
        return classify_count(test_score, thr)
    if qntMethod == "acc":        
        return ACC(test_score, TprFpr)
    #if qntMethod == "emq_old":        
     #   return EMQ(p_score, n_score, test_score)
    #if qntMethod == "emq":        
     #   return EMQ(test_score, tr_dist= 0.95)
    if qntMethod == "emq":        
        return EMQ(test_score, train_labels, nclasses=2)
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
        return GAC(scores, test_score, train_labels, nclasses = 2)
    if qntMethod == "GPAC":
        return GPAC(scores, test_score, train_labels, nclasses = 2)
    #if qntMethod == "FM":
     #   return FM(scores, calib_clf, te_data, train_labels, nclasses = 2)
    if qntMethod == "temp":
        return temporary(scores, test_score, train_labels, nclasses = 2)
    

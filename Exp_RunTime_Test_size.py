import pandas as pd
import numpy as np
import pickle
import time
import os
from applyquantifiers import apply_quantifier
from methodlist import methods
import sys
import argparse
import helpers
from PWKCLF import PWKCLF
from sklearn.model_selection import train_test_split
from schumar_model_fit import fit_quantifier_schumacher_github
import pdb

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def Run_expereiment(method_name, niterations = 100 ):

    folder = "exp_setup_runtime_data"
    result_path = "exp_setup_runtime_data/runtime_results_test_size"
    os.makedirs( result_path , exist_ok=True)   # creating directory for saving the results
    measure     = "topsoe"                   #default measure for DyS
    method_kargs = methods[method_name]["kargs"]
      
    num_features = 50
    #...................Schumacher paper methods..........................
    schumi_quantifiers = ['readme', 'HDx', 'FormanMM', 'FM', 'CDE', 'GPAC', 'GAC'] #quantifiers from schumacher paper
    #......................................................................

    df = pd.DataFrame( np.random.rand(10000,num_features))
    lb = np.random.randint(2,size= 10000)
    lbt = pd.DataFrame(lb.T, columns= ['class'])
    dt = pd.concat([lbt, df], axis= 1)
    dt.columns = dt.columns.astype(str)

    train_dt, test_dt, train_label, test_label = train_test_split(dt, dt["class"], test_size = 0.5, stratify=dt["class"])
    
    train_dt = pd.DataFrame(train_dt)
    #train_dt['class'] = train_dt['class'].astype('str')
    label = train_dt['class']
    instances =train_dt.drop('class', axis=1)
    ##############################################################################
    
    Test_size = [100, 1000, 10000, 20000,30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000 ]
    table= pd.DataFrame( columns =['Quantifier', 'TestSize', 'Runtime'])
    
    for size in Test_size:
        print('Test Size:', size)
        #...............Lodaing Scores, TPRFPR and Models.......................
        scores = pd.read_csv(folder + '/%d' % (num_features) +'/Scores_for_Runtime_Exp.csv', index_col= False)
        tprfpr = pd.read_csv(folder + '/%d' % (num_features) + '/TPRFPR_for_Runtime_Exp.csv', index_col= False)
        rf_clf = pickle.load(open(folder + '/%d' % (num_features) + '/rf_clf_for_Runtime_Exp.pkl', 'rb'))         
        calibrt_clf = pickle.load(open(folder + '/%d' % (num_features) + '/calibrt_clf_for_Runtime_Exp.pkl', 'rb'))
        #pwk_clf = pickle.load(open(folder + '/%d' % (num_features) + '/model_pwk_for_Runtime_Exp.pkl', 'rb'))
        pwk_clf = None

        #.........................Schumacher Methods.............................................................
        schumacher_qnt = None
        
        if method_name in schumi_quantifiers:
            schumacher_qnt = fit_quantifier_schumacher_github(method_name, instances, label)
        #.....................Separating Positive and negative training scores.............................................................

        pos_scores = scores[scores["class"]==1]["scores"]   #separating positve scores from training scores  
        neg_scores = scores[scores["class"]==0]["scores"]

        ################################################################################
        test_dt = pd.DataFrame( np.random.rand(size,num_features))
        test_label = pd.DataFrame(np.random.randint(2,size= size).T , columns= ['class'])
        
        te_scores = rf_clf.predict_proba(test_dt)[:,1]  #estimating test sample scores
        
        tot_time = []
        for it in range(niterations):
            
            print('iteration :', it+1)
            #............Calling Methods for Runtime Estimation.............
            start = time.time()           
            pdb.set_trace()
            #pred_pos_prop = apply_quantifier(qntMethod= method_name,scores=scores,p_score=pos_scores,n_score=neg_scores, test_score=te_scores, TprFpr = tprfpr, thr = 0.5, measure = measure, calib_clf = calibrt_clf ,te_data = test_sample,**method_kargs)
            pred_pos_prop = apply_quantifier(qntMethod= method_name,p_score=pos_scores,n_score=neg_scores, test_score=te_scores, TprFpr = tprfpr, thr = 0.5, measure = measure, calib_clf = calibrt_clf ,te_data = test_dt,pwk_clf= pwk_clf,schumacher_qnt=schumacher_qnt,**method_kargs)
            stop = time.time() 
            print(stop) 
            pdb.set_trace()
            tot_time.append(stop - start)
        mean_time = np.mean(tot_time)
        #table = table.append(pd.DataFrame([method_name,size,mean_time]).T)
        table = table.append({'Quantifier': method_name,'TestSize':size,'Runtime' :mean_time}, ignore_index = True)
        
                
        #table.columns =['Quantifier', 'TestSize', 'Runtime']    
        table.to_csv(result_path + '/'+method_name + '_runtime.csv', index=False)
        print(table) 
        

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  #parser.add_argument('exp', type=str)
  #parser.add_argument('method', type=str)
  #parser.add_argument('--it', type=int, default=100)
  #args = parser.parse_args()
  Run_expereiment('cc', 100)
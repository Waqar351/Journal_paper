import pandas as pd
import numpy as np
import pickle
import time
import os
from applyquantifiers import apply_quantifier
from methodlist import methods
from sklearn.ensemble import RandomForestClassifier
import sys
import argparse
import helpers
from PWKCLF import PWKCLF
from sklearn.model_selection import train_test_split
from schumar_model_fit import fit_quantifier_schumacher_github
import quapy as qp
import sys

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

        tr = pd.DataFrame( np.random.rand(size,num_features))
        tr_label = pd.DataFrame(np.random.randint(2,size= size).T , columns= ['class'])

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

        
        #..................PWK model learning.....................
        model_pwk = None
        if method_name == 'PWK':
            clf=PWKCLF(alpha=1, n_neighbors=10, algorithm="auto",
                            metric="euclidean", leaf_size=30, p=2,
                            metric_params=None, n_jobs=None)

            model_pwk = clf.fit(tr, tr_label)
        
         #........................Training EMQ method using Quapy library....................
        mod_quapy = None    #model quapy
        

        tr_quapy = qp.data.LabelledCollection(tr,tr_label['class'].to_numpy())
       
        if method_name =='emq':
            mod_quapy = qp.method.aggregative.EMQ(RandomForestClassifier(n_estimators=200))
            mod_quapy.fit(tr_quapy)


        
         #..............Test Sample QUAPY exp...........................
        te_quapy = qp.data.LabelledCollection(test_dt, test_label['class'].to_numpy())

        tot_time = []
        for it in range(niterations):
            
            print('iteration :', it+1)
            #............Calling Methods for Runtime Estimation.............       

            #pred_pos_prop = apply_quantifier(qntMethod= method_name,scores=scores,p_score=pos_scores,n_score=neg_scores, test_score=te_scores, TprFpr = tprfpr, thr = 0.5, measure = measure, calib_clf = calibrt_clf ,te_data = test_sample,**method_kargs)
            time_cost = apply_quantifier(qntMethod= method_name, 
                                            scores = scores['scores'], 
                                            p_score=pos_scores,
                                            n_score=neg_scores, 
                                            train_labels = scores['class'], 
                                            test_score = te_scores, 
                                            TprFpr = tprfpr, 
                                            thr = 0.5, 
                                            measure = measure, 
                                            calib_clf = calibrt_clf, 
                                            te_data = test_dt, 
                                            pwk_clf = model_pwk, 
                                            schumacher_qnt = schumacher_qnt, 
                                            test_quapy = te_quapy, 
                                            model_quapy = mod_quapy,  
                                            **method_kargs)
            tot_time.append(time_cost)
        mean_time = np.mean(tot_time)
        #table = table.append(pd.DataFrame([method_name,size,mean_time]).T)
        table = table.append({'Quantifier': method_name,'TestSize':size,'Runtime' :mean_time}, ignore_index = True)
        
                
        #table.columns =['Quantifier', 'TestSize', 'Runtime']    
        table.to_csv(result_path + '/'+method_name + '_runtime.csv', index=False)
    print(table) 
        

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  #parser.add_argument('exp', type=str)
  parser.add_argument('method', type=str)
  parser.add_argument('--it', type=int, default=100)
  args = parser.parse_args()
  #Run_expereiment(sys.argv[1], 100)
  Run_expereiment(args.method, args.it)
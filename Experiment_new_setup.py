from timeit import default_timer as timer
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import argparse
from methodlist import methods
import sys
from applyquantifiers import apply_quantifier
from PWKCLF import PWKCLF
from schumar_model_fit import fit_quantifier_schumacher_github
import os


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def Run_expereiment(exp_name, method_name, niterations = 10 ):
   """
   This is the main function to run the experimental setup. It returns performance of quantification methods in terms MAE across different test sample sizes and training proportions and save the results in .CSV file.
    
   Parameters
   ---------
   exp_name : string
       name of a dataset (user input)
   method_name : string
       name of the method (user input)
   niterations : integer
        number of iterations to repeat the experiment

    Returns
    ---------
    CSV file
        that contains performance of Quantification method in terms of MAE.
    
   """

    #>>>>>>>..............Experimental_setup............>>>>>>>>>>
   vdist = ["topsoe", "jensen_difference", "prob_symm", "ord", "sord", "hellinger"] 
   names_vdist = ["TS", "JD", "PS", "ORD", "SORD", "HD"] 
   counters    = ["HDy","DyS-TS","SORD", "MS", "CC", "ACC","SMM"]
   measure     = "topsoe"                   #default measure for DyS
   method_kargs = methods[method_name]["kargs"]

   result_path = "exp2_results"                 #Saving the output
   os.makedirs( result_path , exist_ok=True) 

   table2=pd.DataFrame() 
   
   tr_prop = [0.05,0.1,0.3,0.5,0.7,0.9]         # Training proportions used to train the classifier
   tr_sample= 10                                # Number of Iterations for training proportion
   for train_prop in tr_prop:
       for train_sample in range(tr_sample):

            folder = 'models_new_exp_setup_2/'+ exp_name +'/%d' % (train_sample+1) + '/%f' % train_prop 
            tr = pd.read_csv(folder + '/train_data_%s' % exp_name + '_%f'%train_prop + '.csv', index_col=False, engine='python')
            te = pd.read_csv(folder + '/test_data_%s' % exp_name + '_%f'%train_prop + '.csv', index_col=False, engine='python')
            tprfpr = pd.read_csv(folder + '/tprfpr_%s'% exp_name + '_%f'%train_prop + '.csv', index_col = False, engine='python')
            scores = pd.read_csv(folder +'/scores_training_%s' % exp_name + '_%f'%train_prop +'.csv', index_col=False, engine='python')
            rf_clf = pickle.load(open(folder +'/model_%s' % exp_name +'_%f'%train_prop + '.pkl', 'rb')) 

            #..................Schumacher Paper Methods....................
            #schumi_quantifiers = ['readme', 'HDx', 'FormanMM', 'FM', 'CDE', 'GPAC', 'GAC']
            schumi_quantifiers = ['readme', 'HDx', 'FormanMM', 'CDE']
            
            schumacher_qnt = None
        
            if method_name in schumi_quantifiers:
                schumacher_qnt = fit_quantifier_schumacher_github(method_name, tr.drop(["class","Binary_label"], axis=1), tr['class'])

            #..................PWK model learning.....................
            model_pwk = None
            if method_name == 'PWK':
                clf=PWKCLF(alpha=1, n_neighbors=10, algorithm="auto",
                            metric="euclidean", leaf_size=30, p=2,
                            metric_params=None, n_jobs=None)

                model_pwk = clf.fit(tr.drop(["class","Binary_label"], axis=1), tr['class'])
            
            #..........................calibrated_model_for PCC and PACC..................
            #tr = pd.DataFrame(tr)
            calibrt_clf = None
            if method_name == 'pcc' or 'pacc':
                x_model_train, x_valid, y_model_train, y_valid = train_test_split(tr.drop(["class","Binary_label"], axis =1), tr["Binary_label"], test_size = 0.5, stratify=tr["Binary_label"]) 

                rf_clf2 = RandomForestClassifier(n_estimators=200)
                rf_clf2.fit(x_model_train, y_model_train)         #model is trained on new training set 
                
                calibrt_clf = CalibratedClassifierCV(rf_clf2, method="sigmoid", cv="prefit") #calibrated prbabilities
                calibrt_clf.fit(x_valid, y_valid)
                
            #>>>>>>>>>>>>>>>>................................>>>>>>>>>>>>>>>>>>>>

            pos_scores = scores[scores["class"]==1]["scores"]   #separating positve scores from training scores  
            neg_scores = scores[scores["class"]==0]["scores"]
            
            df_test = pd.DataFrame(te)
            print('test_lenght',len(df_test))
            
            df_test_pos = df_test.loc[df_test["Binary_label"] == 1] # seperating positive test examples
            df_test_neg = df_test.loc[df_test["Binary_label"] == 0] # seperating negative test examples
            
            max_allowed = min(len(df_test_pos), len(df_test_neg))
            batch_sizes = list(range(10, min(91, max_allowed + 1), 10)) + list(range(100, min(501, max_allowed + 1), 100))
            
            alpha_values = [0, 0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]   #Test class proportion
            #alpha_values = [0, 0.01,0.5,0.9,1]   #Test class proportion
            table=pd.DataFrame()
            for sample_size in batch_sizes:   #[10,100,500], batch_sizes, Varying test set sizes
                
                for alpha in alpha_values: #   Varying positive class distribution
                    abs_error = []
                    error = []
                    n_pos_pred = []; pred_pos_prop=[]
                    ms_per_example = []
                    for iter in range(niterations):
                        print('Sample size #%d' % (sample_size))
                        print('iteration #%d' % (iter + 1))

                        pos_size = np.int(round(sample_size * alpha, 2))
                        #neg_size = round(sample_size * (1-alpha),2)
                        neg_size = sample_size - pos_size

                        sample_test_pos = df_test_pos.sample( int(pos_size), replace = False)
                        sample_test_neg = df_test_neg.sample( int(neg_size), replace = False)
                        
                        sample_test = pd.concat([sample_test_pos, sample_test_neg])
                        #print ('Te(+) :', len(sample_test_pos), 'Te(-) :', len(sample_test_neg),'Te :', len(sample_test))
                        
                        test_label = sample_test["Binary_label"]
                        
                        test_sample = sample_test.drop(["class","Binary_label"], axis=1)  #dropping class label columns
                        te_scores = rf_clf.predict_proba(test_sample)[:,1]  #estimating test sample scores
                        
                        n_pos_sample_test = list(test_label).count(1) #Counting num of actual positives in test sample
                        calcultd_pos_prop = round(n_pos_sample_test/len(sample_test), 2) #actual pos class prevalence in generated sample
                    
                        
                        for co in counters:
                            aux = co.split("-")
                            quantifier = co
                            if len(aux) > 1:
                                quantifier = aux[0]
                                measure = vdist[names_vdist.index(aux[1])]
                        
                        #.............Calling of Methods.................................................. 
                        tm_start = timer()
                        pred_pos_prop = apply_quantifier(qntMethod= method_name, scores = scores['scores'], p_score=pos_scores,n_score=neg_scores, train_labels = scores['class'], test_score = te_scores, TprFpr = tprfpr, thr = 0.5, measure = measure, calib_clf = calibrt_clf , te_data = test_sample, pwk_clf = model_pwk, schumacher_qnt = schumacher_qnt, **method_kargs)
                        #pred_pos_prop = apply_quantifier(qntMethod= method_name,p_score=pos_scores,n_score=neg_scores, test_score=te_scores, TprFpr = tprfpr, thr = 0.5, measure = measure, calib_clf = calibrt_clf ,te_data = test_sample)
                        tm_end = timer()

                        pred_pos_prop = np.round(pred_pos_prop,2)  #predicted class proportion
                        
                        #..............................RESULTS Evaluation.....................................

                        print("Calculated pos proportion: ", calcultd_pos_prop)
                        print("Num_actual_Positives:",n_pos_sample_test,"actual_pos_proportion:", alpha )
                        print ("Train_Proportion:",train_prop,(train_sample+1),"predict_pos_proportion:", pred_pos_prop )
                        
                        abs_error = np.round(abs(calcultd_pos_prop - pred_pos_prop),2) #absolute error
                        error = np.round(calcultd_pos_prop - pred_pos_prop , 2)     # simple error Biasness
                        ms_per_example = (tm_end - tm_start) * 1000 / len(sample_test) #time calculation
                        
                        table = table.append(pd.DataFrame([(train_sample+1),train_prop,(iter + 1),sample_size,alpha,calcultd_pos_prop,pred_pos_prop,abs_error,error,ms_per_example,method_name,exp_name]).T)

            table.columns = ["Train_sample","Train_prop","Test_sample#","Test_size","alpha","actual_prop","pred_prop","abs_error","error/bias","time","quantifier","dataset"]
            table2 = table2.append(table)
            
    #table2.to_csv('exp2_results/' + exp_name + '_%f'% train_prop + '_%s' % method_name + '.csv', index = False)
   table2.to_csv(result_path +'/' + exp_name + '_%s' % method_name + '.csv', index = False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('exp', type=str)
  parser.add_argument('method', type=str)
  parser.add_argument('--it', type=int, default=10)
  args = parser.parse_args()
  Run_expereiment(args.exp, args.method, args.it)
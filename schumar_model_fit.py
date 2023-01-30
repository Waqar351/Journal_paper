import pandas as pd
import numpy as np
import helpers
import time
import pdb


def fit_quantifier_schumacher_github(qntMethod, X_train, y_train):
    """This function fit a quantifier using the codes provided by Tobias Schumacher.
 
    Parameters
    ----------
    qntMethod : string
        Quantification method name, according to the alg_index.csv file
    X_train : DataFrame
        A DataFrame of the training data.
    y_train : DataFrame
        A DataFrame with the training labels.
    Returns
    -------
    object
        the quantifier fitted. 
    """

    algorithm_index = pd.read_csv("alg_index.csv",
                                sep=";",
                                index_col="algorithm")

    algorithm_index = algorithm_index.loc[algorithm_index.export == 1]
    algorithms = list(algorithm_index.index)

    algorithm_dict = dict({alg: helpers.load_class(algorithm_index.loc[alg, "module_name"],
                                                algorithm_index.loc[alg, "class_name"])
                        for alg in algorithms})

    init_args = []

    fit_args = [np.asarray(X_train), np.asarray(y_train)]   
    qf = algorithm_dict[qntMethod](*init_args)

    qf.fit(*fit_args)

    return qf  


def predict_quantifier_schumacher_github(qnt, X_test):
    """This function predict the class distribution from a given test set.
 
    Parameters
    ----------
    qnt : object
        A quantifier previously fitted from some training data.
    X_test : DataFrame
        A DataFrame of the test data.
    Returns
    -------
    array
        the class distribution of the test calculated according to the qntMethod quantifier. 
    """

    start = time.time()
    re = qnt.predict(*[np.asarray(X_test)])
    stop = time.time()
    #return stop - start
    return re
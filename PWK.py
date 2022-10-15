import numpy as np
import pandas as pd

def PWK(test, clf):
    #test['class'] = test['class'].astype('str')
    #test = pd.DataFrame(test.drop(['class'], axis=1))
    test = pd.DataFrame(test)
    #print(test)
    #exit()
    #test = test.reshape(1, -1)
    #_, pred = np.unique(clf.predict(test.drop(['class'], axis=1)), return_counts=True)
    _, pred = np.unique(clf.predict(test), return_counts=True)

    prop = pred/sum(pred)
    #print(prop[0]) 
    prop = np.array(prop[0])
    #exit()
    return prop
import time


def EMQ_quapy(test, model):
    
    start = time.time()
    pos_prop = model.quantify(test.instances)
    stop = time.time()
    #return stop - start
    prev = pos_prop[1]
    
    return prev


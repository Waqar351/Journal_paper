
def EMQ_quapy(test, model):
    
    pos_prop = model.quantify(test.instances)
    
    prev = pos_prop[1]
    
    return prev


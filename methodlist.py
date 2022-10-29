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
from emq_quapy import EMQ_quapy
from QT import QT
from PWK import PWK
from QT_ACC import QT_ACC
from GAC import GAC
from GPAC import GPAC
from FM import FM
from schumar_model_fit import predict_quantifier_schumacher_github

methods = {
  "cc": {
    "func": classify_count,
    "kargs": {},
  },
  
  "acc": {
    "func": ACC,
    "kargs": {},
  },
  "pcc": {
    "func": PCC,
    "kargs": {},
  },

  "pacc": {
    "func": PACC,
    "kargs": {},
  },
  "hdy": {
    "func": Hdy,
    "kargs": {},
  },
  "ms": {
    "func": MS_method,
    "kargs": {}
  },

  "ms2": {
    "func": MS_method2,
    "kargs": {}
  },
  "x": {
    "func": X,
    "kargs": {}
  },
  "max": {
    "func": Max,
    "kargs": {}
  },
  "t50": {
    "func": T50 ,
    "kargs": {}
  },
  "smm": {
    "func": SMM,
    "kargs": {}
  },

  "dys_Ts": {
    "func": dys_method,
    "kargs": {}
  },
  "sord": {
    "func": SORD_method,
    "kargs": {}
  },
  #"emq": {
   # "func": EMQ,
    #"kargs": {}
  #},
  "emq": {
    "func": EMQ_quapy,
    "kargs": {}
  },
  "QT": {
    "func": QT,
    "kargs": {}
  },
  "PWK": {
    "func": PWK,
    "kargs": {}
  },
  "QT_ACC": {
    "func": QT_ACC,
    "kargs": {}
  },
  "readme": {
    "func": predict_quantifier_schumacher_github,
    "kargs": {}
  },
  "FM": {
    "func": predict_quantifier_schumacher_github,
    "kargs": {}
  },
  "CDE": {
    "func": predict_quantifier_schumacher_github,
    "kargs": {}
  },
  "HDx": {
    "func": predict_quantifier_schumacher_github,
    "kargs": {}
  },
  "GPAC": {
    "func": predict_quantifier_schumacher_github,
    "kargs": {}
  },
  "GAC": {
    "func": predict_quantifier_schumacher_github,
    "kargs": {}
  },
  "FormanMM": {
    "func": predict_quantifier_schumacher_github,
    "kargs": {}
  },
  "EM": {
    "func": predict_quantifier_schumacher_github,
    "kargs": {}
  },
  
  #"GAC": {
   # "func": GAC,
    #"kargs": {}
  #},
  #"GPAC": {
   # "func": GPAC,
    #"kargs": {}
  #},
  #"FM": {
   # "func": FM,
    #"kargs": {}
  #},

}
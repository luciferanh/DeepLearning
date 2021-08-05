import numpy as np
from math import *

scores=[-3.44,1.16,3.91]

s_exp = [exp(i) for i in scores]

P_i = [es/(sum(s_exp)) for es in s_exp]

log_P=[-log(p) for p in P_i]
print(log_P)

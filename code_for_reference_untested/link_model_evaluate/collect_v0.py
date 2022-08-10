import json 
import numpy as np 

testing_set = [0,5,6,11,12,17,18,22,25,28,31]

result = []
for tid in testing_set:
    result += json.load(open("results/v0_ret_%d.json" % tid))

print([np.mean([item[k] for item in result]) for k in range(4)])
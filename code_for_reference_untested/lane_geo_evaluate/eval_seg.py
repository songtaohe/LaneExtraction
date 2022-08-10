from eval import Evaluator
import numpy as np 

matched_c, prop_c, gt_c = 0, 0, 0
precisions, recalls = [], []
for tid in [0,5,6,11,12,17,18,22,25,28,31]:
    e = Evaluator()
    e.loadGTFromWays("../dataset_evaluation/way_%d.json" % tid)
    e.loadPropFromGraph("../all_results2/%d/graph.p" % tid)
    matched_n, prop_n, gt_n = e.geoMetric(mask = "../dataset_evaluation/regionmask_%d.jpg" % tid)   

   

    precision = float(matched_n) / prop_n
    recall = float(matched_n) / gt_n

    precisions.append(precision)
    recalls.append(recall)

    print(matched_n, prop_n, gt_n, precision, recall)

print(precisions)
print(recalls)

print(np.mean(precisions))
print(np.mean(recalls))

from eval import Evaluator
import numpy as np 
import sys 
import json 

matched_c, prop_c, gt_c = 0, 0, 0
geo_precisions, geo_recalls = [], []
topo_precisions, topo_recalls = [], []

for tid in [0,5,6,11,12,17,18,22,25,28,31]:
    e = Evaluator()
    e.loadGTFromWays("../dataset_evaluation/way_%d.json" % tid)

    if sys.argv[1] == "sat2graph":
        e.loadPropFromGraph("../lane_model_sat2graph/result/%d/graph_output.png_graph.p" % tid)
    else:
        e.loadPropFromGraph("../all_results_%s/%d/graph.p" % (sys.argv[1], tid))
    #e.loadPropFromGraph("../lane_model_sat2graph/result/%d/graph_output.png_graph.p" % tid)
    
    geo_p, geo_r, topo_p, topo_r = e.topoMetric(mask = "../dataset_evaluation/regionmask_%d.jpg" % tid)   

    geo_precisions.append(geo_p)
    geo_recalls.append(geo_r)

    topo_precisions.append(topo_p * geo_p)
    topo_recalls.append(topo_r * geo_r)


# TODO save the result


print("geo precision", np.mean(geo_precisions))
print("geo recall", np.mean(geo_recalls))

print("topo precision", np.mean(topo_precisions))
print("topo recall", np.mean(topo_recalls))

ret = {}
ret["geo precision"] = np.mean(geo_precisions)
ret["geo recall"] = np.mean(geo_recalls)
ret["topo precision"] = np.mean(topo_precisions)
ret["topo recall"] = np.mean(topo_recalls)

json.dump(ret, open(sys.argv[1]+".json", "w"), indent=2)
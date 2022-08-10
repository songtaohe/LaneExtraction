import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from PIL import Image 
import numpy as np 
import scipy.misc
import scipy.ndimage
import json 
import cv2
import pickle
from hdmapeditor.roadstructure import LaneMap 
from time import time 
from segtograph.segtographfunc import segtograph
import math

from evaluate import Evaluator


if __name__ == "__main__":
	evaluator = Evaluator(sys.argv[1], sys.argv[2])
	from hdmapv3lane.infer_link_engine import InferEngine

	inferEngine = InferEngine(batchsize=4)

	result = []
	cc = 0
	direction_img = np.zeros((640, 640, 3))

	newbatch3 = np.zeros((4, 640, 640, 7)) - 0.5

	while True:
		batch = evaluator.loadbatch(batchsize = 4)
		if batch[0] == 0:
			break
		
		ret = inferEngine.infer(sat = batch[2], connector = batch[3], direction = batch[4])
		#newbatch3[:,:,:,0:3] = batch[3][:,:,:,3:6]
		#newbatch3[:,:,:,3:6] = batch[3][:,:,:,0:3]
		#ret2 = inferEngine.infer(sat = batch[2], connector = newbatch3, direction = batch[4])
	
		

		
		for i in range(batch[0]):
			#pred = max(ret[i,0], ret2[i,0])
			pred = ret[i,0]
			if pred > 0.5:
				result.append(1)
			else:
				result.append(0)


			# Image.fromarray(((batch[2][i,:,:,:] + 0.5) * 255).astype(np.uint8) ).save("debug/input%d.jpg" % (cc))
			# Image.fromarray(((batch[3][i,:,:,0:3]) * 127 + 127).astype(np.uint8) ).save("debug/connector1%d.jpg" % (cc))
			# Image.fromarray(((batch[3][i,:,:,3:6]) * 127 + 127).astype(np.uint8) ).save("debug/connector2%d.jpg" % (cc))

			# if (evaluator.labels[cc] > 0.5 and pred > 0.5) or (evaluator.labels[cc] <= 0.5 and pred <= 0.5):
			# 	correct = 1
			# else:
			# 	correct = 0

			# with open("debug/label%d.txt" % (cc), "w") as fout:
			# 	fout.write("%f %f %d\n" % (evaluator.labels[cc], ret[i,0], correct))

			# direction_img[:,:,2] = np.clip(batch[4][i,:,:,0],-1,1) * 127 + 127
			# direction_img[:,:,1] = np.clip(batch[4][i,:,:,1],-1,1) * 127 + 127
			# direction_img[:,:,0] = 127

			# direction_img[:,:,0] += batch[3][i,:,:,0] * 255 + 127
			# direction_img[:,:,1] += batch[3][i,:,:,3] * 255 + 127
			# direction_img[:,:,2] += batch[3][i,:,:,6] * 255 + 127
			
			# direction_img = np.clip(direction_img, 0, 255)

			# Image.fromarray(direction_img.astype(np.uint8) ).save("debug/direction%d.jpg" % (cc))


			cc += 1

	json.dump([evaluator.labels, result], open("results/v1_ret%s.json" % sys.argv[2], "w"), indent=2)
	evaluator.checkResult(result)
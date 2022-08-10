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

from hdmapv3lane.lane_geo_evaluate.eval import Evaluator as topoEvaluator

class Evaluator():
	def __init__(self, dataset_folder, tileid):
		sat_img = scipy.ndimage.imread(dataset_folder+"/sat%s.jpg" % tileid)
		mask = scipy.ndimage.imread(dataset_folder+"/regionmask%s.jpg" % tileid)
		direction = scipy.ndimage.imread(dataset_folder+"/normal%s.jpg" % tileid)
		links = json.load(open(dataset_folder+"/link%s.json" % tileid))

		nidmap, nodes, locallinks = links 

		start_nodes = set()
		end_nodes = set()

		for locallink in locallinks:
			start_nodes.add(tuple(locallink[0]))
			end_nodes.add(tuple(locallink[-1]))

			

		print("number of terminal nodes", len(nidmap))

		pairs = []
		labels = []
		# for nid1 in nidmap.keys():
		# 	for nid2 in nidmap.keys():
		# 		# if nid1 >= nid2 :
		# 		# 	continue
		# 		if nid1 not in nodes or nid2 not in nodes:
		# 			continue

		# 		if tuple(nodes[nid1]) in start_nodes and tuple(nodes[nid2]) in end_nodes:
		# 			pass
		# 		else:
		# 			continue
				
		# 		r = 70 * 8
		# 		if (nodes[nid1][0] - nodes[nid2][0])**2 + (nodes[nid1][1] - nodes[nid2][1])**2 < r**2:
		# 			pairs.append((nid1, nid2))

		# 			if int(nid1) in nidmap[nid2] or int(nid2) in nidmap[nid1]:
		# 				labels.append(1)
		# 			else:
		# 				labels.append(0)

		for locallink in locallinks:
			nid1 = tuple(locallink[0])
			nid2 = tuple(locallink[-1])
			
			pairs.append((nid1, nid2))
			labels.append(1)


		#print(nidmap)

		print("number of candidate pairs", len(pairs), np.sum(labels))
		#exit()

		padding_size = 320 

		self.sat = np.pad(sat_img, [[padding_size,padding_size],[padding_size,padding_size],[0,0]], 'constant').astype(np.float) / 255.0 - 0.5
		self.sat = self.sat * 0.81
		self.mask = np.pad(mask, [[padding_size,padding_size],[padding_size,padding_size],[0,0]], 'constant')
		
		direction = (direction.astype(np.float) - 127) / 127.0
		direction = direction[:,:,1:3]

		self.direction = np.zeros((4096, 4096, 2))
		self.direction[:,:,0] = direction[:,:,1]
		self.direction[:,:,1] = direction[:,:,0]
		self.direction = np.pad(self.direction, [[padding_size,padding_size],[padding_size,padding_size],[0,0]], 'constant')
		
		self.links = links 
		self.pairs = pairs 
		self.labels = labels
		self.ptr = 0 

		self.dataset_folder = dataset_folder
		self.tileid = tileid 
		
		image_size = 640 

		self.maxbatchsize = 32
		self.image_batch = np.zeros((self.maxbatchsize, image_size, image_size,3))
		self.direction_batch = np.zeros((self.maxbatchsize, image_size, image_size,2))
		self.connector_batch = np.zeros((self.maxbatchsize, image_size, image_size,7))
		
		self.poscode = np.zeros((image_size * 2, image_size * 2,2))
		for i in range(image_size * 2):
			self.poscode[i,:,0] = float(i) / image_size - 1.0
			self.poscode[:,i,1] = float(i) / image_size - 1.0

		self.padding_size = padding_size
		self.image_size = image_size
		self.t0 = time()

	def loadbatch(self, batchsize):
		img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
		if self.ptr >= len(self.pairs):
			return 0, None

		p = int(float(self.ptr) / len(self.pairs) * 60)
		sys.stdout.write("\rprogress " + "|" * p + '.' * (60-p) + " %d/%d time %.2f" % (self.ptr, len(self.pairs), time() - self.t0))
		sys.stdout.flush()	

		cc = 0
		positions = []
		batch_links = []

		for i in range(batchsize):
			ptr = self.ptr + i 
			if ptr >= len(self.pairs):
				break 
			cc += 1
			nid1, nid2 = self.pairs[ptr]
			pos1 = [nid1[0] + self.padding_size, nid1[1] + self.padding_size] 
			pos2 = [nid2[0] + self.padding_size, nid2[1] + self.padding_size] 
			
			start = [(pos1[0] + pos2[0])//2 - self.image_size//2, (pos1[1] + pos2[1])//2 - self.image_size//2]

			self.image_batch[i,:,:,:] = self.sat[start[1]:start[1] + self.image_size, start[0]:start[0] + self.image_size, :]
			self.direction_batch[i,:,:,:] = self.direction[start[1]:start[1] + self.image_size, start[0]:start[0] + self.image_size, :]
			img *= 0
			cv2.circle(img, (pos1[0] - start[0], pos1[1] - start[1]), 12, (255), -1)
			self.connector_batch[i,:,:,0] = np.copy(img) / 255.0 - 0.5

			img *= 0
			cv2.circle(img, (pos2[0] - start[0], pos2[1] - start[1]), 12, (255), -1)
			self.connector_batch[i,:,:,3] = np.copy(img) / 255.0 - 0.5

			x1 = pos1[0] - start[0]
			y1 = pos1[1] - start[1]

			x2 = pos2[0] - start[0]
			y2 = pos2[1] - start[1]

			positions.append([[x1,y1], [x2,y2]])

			self.connector_batch[i,:,:,1:3] = self.poscode[self.image_size - y1:self.image_size*2 - y1, self.image_size - x1:self.image_size*2 - x1, :]
			self.connector_batch[i,:,:,4:6] = self.poscode[self.image_size - y2:self.image_size*2 - y2, self.image_size - x2:self.image_size*2 - x2, :]
			self.connector_batch[i,:,:,6] = -0.5

			locallink = self.links[2][ptr]

			for j in range(len(locallink)):
				locallink[j][0] -= start[0] - self.padding_size
				locallink[j][1] -= start[1] - self.padding_size
			
			batch_links.append(locallink)

		self.ptr += cc 
		#print(self.ptr)
		st = 0			
		return cc, positions, self.image_batch[st:st+batchsize, :,:,:], self.connector_batch[st:st+batchsize,:,:,:], self.direction_batch[st:st+batchsize,:,:,:], batch_links

	def checkResult(self, prediction):
		prediction = [1 if x > 0.5 else 0 for x in prediction]

		correct_n = 0 
		union_n = 0 
		#intersection_n = 0 
		correct_pos_n = 0
		label_pos_n = 0
		pred_pos_n = 0 

		for i in range(len(prediction)):
			if prediction[i] == self.labels[i]:
				correct_n += 1

			if prediction[i] == self.labels[i] and prediction[i] == 1:
				correct_pos_n += 1

			if prediction[i] + self.labels[i] >= 1:
				union_n += 1

			if prediction[i] == 1:
				pred_pos_n += 1
			if self.labels[i] == 1:
				label_pos_n += 1
		
		print(correct_n, correct_pos_n, union_n, label_pos_n, pred_pos_n)

		print("Accuracy ", float(correct_n) / len(prediction))
		print("Precision ", float(correct_pos_n) / (pred_pos_n + 0.001) )
		print("Recall ", float(correct_pos_n) / (label_pos_n + 0.001) )

		p = float(correct_pos_n) / (pred_pos_n + 0.001)
		r = float(correct_pos_n) / (label_pos_n + 0.001)

		print("F1", 2*p*r/(p+r+0.0001))

		print("IoU", float(correct_pos_n) / union_n)




		pass


	def seg2link(self, seg, p1, p2):
		p1 = (p1[1], p1[0])
		p2 = (p2[1], p2[0])
		#print(p1,p2)
		graph = segtograph((seg * 255).astype(np.uint8), 127, 0, 0, 32)
		#print("graph", graph)

		if len(graph) == 0:
			return False, None 
		
		def distance(pos1, pos2):
			a = pos1[0] - pos2[0]
			b = pos1[1] - pos2[1]

			return math.sqrt(a*a + b*b)

		p1dist = 100
		p2dist = 100
		p1nid = None  
		p2nid = None 

		for nid, nei in graph.items():
			if len(nei) == 1:
				d1 = distance(nid, p1)
				if d1 < p1dist:
					p1dist = d1
					p1nid = nid 
				
				d2 = distance(nid, p2)
				if d2 < p2dist:
					p2dist = d2
					p2nid = nid 

		if p1nid is None or p2nid is None:
			return False, None 
		if p1nid == p2nid:
			return False, None 

		#print(p1nid, p2nid)

		newgraph = {}
		for nid, nei in graph.items():
			newnid = nid
			# print(type(newnid), newnid)
			# print(type(p1nid), p1nid)
			if newnid == p1nid:
				newnid = p1
			elif newnid == p2nid:
				newnid = p2 
			newnei = []
			for nn in nei:
				new_nn = nn 
				if new_nn == p1nid:
					new_nn = p1
				elif new_nn == p2nid:
					new_nn = p2

				newnei.append(new_nn)
			newgraph[newnid] = newnei 

		# find a path 
		link = []
		cur = p1
		failed = False
		while True:
			#print(cur, p1, p2)
			link.append(cur)
			if cur == p2:
				break
			nei = newgraph[cur]
			if len(link) == 1:
				cur = nei[0]
				if len(nei) > 1:
					failed = True
					break
			else:
				if len(nei) != 2:
					failed = True
					break
				
				if nei[0] == link[-2]:
					cur = nei[1]
				else:
					cur = nei[0]

		return not failed, link	
		
		

if __name__ == "__main__":
	evaluator = Evaluator(sys.argv[1], sys.argv[2])
	from hdmapv3lane.link_model_v0.infer import InferEngine

	inferEngine = InferEngine(batchsize=4)

	result = []
	
	while True:
		batch = evaluator.loadbatch(batchsize = 4)
		if batch[0] == 0:
			break
		
		ret = inferEngine.infer(sat = batch[2], connector = batch[3], direction = batch[4])
		for i in range(batch[0]):
			r = evaluator.seg2link(ret[i,:,:,0], batch[1][i][0], batch[1][i][1])
			print(r)
			print(batch[5][i])

			if r[0] == True:
				gt_graph = {}
				locallink = batch[5][i]
				for j in range(len(locallink) - 1):
					nk1 = (locallink[j][1], locallink[j][0]) 
					nk2 = (locallink[j+1][1], locallink[j+1][0])

					if nk1 not in gt_graph:
						gt_graph[nk1] = [nk2] 
					elif nk2 not in gt_graph[nk1]:
						gt_graph[nk1].append(nk2)

					if nk2 not in gt_graph:
						gt_graph[nk2] = [nk1] 
					elif nk1 not in gt_graph[nk2]:
						gt_graph[nk2].append(nk1)

				prop_graph = {}
				link = r[1]
				for j in range(len(link)-1):
					nk1 = (link[j][0], link[j][1]) 
					nk2 = (link[j+1][0], link[j+1][1])

					if nk1 not in prop_graph:
						prop_graph[nk1] = [nk2] 
					elif nk2 not in prop_graph[nk1]:
						prop_graph[nk1].append(nk2)

					if nk2 not in prop_graph:
						prop_graph[nk2] = [nk1] 
					elif nk1 not in prop_graph[nk2]:
						prop_graph[nk2].append(nk1)

				
				e = topoEvaluator()
				e.setGT(gt_graph)
				e.setProp(prop_graph)
				geo_p, geo_r, topo_p, topo_r = e.topoMetric(verbose = False)
				print(geo_p, geo_r, topo_p, topo_r)   

				result.append((geo_p, geo_r, topo_p, topo_r))

				print([np.mean([item[k] for item in result]) for k in range(4)])

			#print(len(result), r)
			# Image.fromarray((ret[i,:,:,0] * 255).astype(np.uint8)).save("debug/pair%d_output.png" % (len(result)))

			# img = ((batch[2][i,:,:,0:3] + 0.5) * 255).astype(np.uint8)

			# color = (255,0,0)
			# if evaluator.labels[len(result)] == 1:
			# 	color = (0,255,0)

			# cv2.circle(img, tuple(batch[1][i][0]), 5, color,-1)
			# cv2.circle(img, tuple(batch[1][i][1]), 5, color,-1)
			
			# if r[0] == True:
			# 	# link = r[1]
			# 	# for j in range(len(link)-1):
			# 	# 	y1,x1 = link[j]
			# 	# 	y2,x2 = link[j+1]
			# 	# 	cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 3)

			# 	result.append(1)

			# else:
			# 	result.append(0)

		#exit()

			# Image.fromarray(img).save("debug/pair%d_output_sat.png" % (len(result)-1))
				
			# if len(result) > 10:
			# 	exit()
	
	
	json.dump(result, open("results/v0_ret%s.json" % sys.argv[2], "w"), indent=2)
	# evaluator.checkResult(result)
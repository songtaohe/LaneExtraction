import sys 
import json 
import pickle 
import numpy as np 
import rtree 
import cv2 
import scipy.ndimage
from time import time 

class Evaluator():
	def __init__(self):
		pass 

	def loadGTFromWays(self, name):
		data = json.load(open(name, "r"))
		neighbors = {}
		for way in data:
			for i in range(len(way)-1):
				x1,y1 = way[i][1], way[i][0]
				x2,y2 = way[i+1][1], way[i+1][0]

				k1 = (x1,y1)
				k2 = (x2,y2)

				if k1 not in neighbors:
					neighbors[k1] = []

				if k2 not in neighbors[k1]:
					neighbors[k1].append(k2)

		self.gt_graph = neighbors

	def loadPropFromGraph(self,name):
		self.prop_graph = pickle.load(open(name, 'r'))

	# new data structure of graph
	# including: 
	#   neighbors the same as graph
	#   node direction on interpolated graph
	# need to add new graph loading function
	# add random noise on interpolated nodes to avoid different nodes having same coordinates

	def loadGraphV2(self, wayname, linkname, interpolate = 2, addnoise = True, transpose = False):
		neighbors = {}
		node_directions = {}
		direction_pairs = set()

		# gt and prop should have the same format
		waydata = json.load(open(wayname, "r"))

		# gt and prop have different formats
		if linkname is not None:
			linkdata = json.load(open(linkname, "r"))

			if len(linkdata) == 3:
				try:
					print(linkdata[2][0][0])
					linkdata = linkdata[2]
				except:
					pass
		else:
			linkdata = None 

		
		if linkdata is not None:
			waydata = waydata + linkdata


		if transpose:
			for way in waydata:
				for i in range(len(way)):
					way[i][0], way[i][1] = way[i][1], way[i][0]

		for way in waydata:
			# add small noise
			if addnoise == True:
				for i in range(1,len(way)-1):
					way[i][0] += (np.random.random() - 0.5) * 0.001
					way[i][1] += (np.random.random() - 0.5) * 0.001
					
			for i in range(len(way)-1):
				x1,y1 = way[i][1], way[i][0]
				x2,y2 = way[i+1][1], way[i+1][0]

				L = int(np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)) // interpolate + 1
				L = max(2,L)
				last_node = (x1,y1)
				for j in range(1,L):
					a = 1.0 - float(j) / (L-1)
					x = x1*a + x2*(1-a)
					y = y1*a + y2*(1-a)
					
					nk1 = last_node
					nk2 = (x, y)
					
					if x < 0 or x >= 4096 or y < 0 or y >= 4096:
						last_node = (x, y)
						continue
					
					if last_node[0] < 0 or last_node[0] >= 4096 or last_node[1] < 0 or last_node[1] >= 4096:
						last_node = (x, y)
						continue
					
					if nk1 not in neighbors:
						neighbors[nk1] = [nk2]
					elif nk2 not in neighbors[nk1]:
						neighbors[nk1].append(nk2)

					if nk2 not in neighbors:
						neighbors[nk2] = [nk1]
					elif nk1 not in neighbors[nk2]:
						neighbors[nk2].append(nk1)

					direction_pairs.add((nk1, nk2))

					last_node = (x,y)

			
		
		for nid, nei in neighbors.items():
			if len(nei) <= 2:
				vx,vy = 0,0
				for nn in nei:
					if (nid,nn) in direction_pairs:
						vx += nn[0] - nid[0]
						vy += nn[1] - nid[1]
					elif (nn,nid) in direction_pairs:
						vx -= nn[0] - nid[0]
						vy -= nn[1] - nid[1]

				
				l = np.sqrt(vx**2 + vy**2)
				vx = vx / l 
				vy = vy / l 

				node_directions[nid] = (vx, vy)

		return neighbors, node_directions


	def visGraph(self, filename, neighbors, node_directions):
		img = np.zeros((4096, 4096, 3), dtype=np.uint8)
		for nid, nv in node_directions.items():
			x = int(nid[0])
			y = int(nid[1])

			color = (int(nv[0] * 127 + 127), int(nv[1] * 127 + 127), 127)

			cv2.circle(img, (x,y), 3, color, -1)

		cv2.imwrite(filename, img)


	# def getNodesFromGraph(self, graph): # place nodes every 0.25 meter 
	# 	nodes = set()
	# 	exist = set()
	# 	for nid, nei in graph.items():
	# 		for nn in nei:
	# 			if (nid, nn) in exist or (nn, nid) in exist:
	# 				continue
	# 			x1,y1 = nid 
	# 			x2,y2 = nn 

	# 			exist.add((nid, nn))

	# 			L = int(np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)) // 2 + 1
	# 			L = max(2,L)
	# 			for i in range(L):
	# 				a = float(i) / (L-1)
	# 				x = x1*a + x2*(1-a)
	# 				y = y1*a + y2*(1-a)

	# 				if x < 0 or x >= 4096 or y < 0 or y >= 4096:
	# 					continue

	# 				if (x,y) not in nodes:
	# 					nodes.add((x,y))

	# 	return list(nodes)

	
	def propagate(self, graph, nid, steps = 200):
		visited = set()
		queue = [(nid,0)]
		while len(queue) > 0:
			cur_nid, depth = queue.pop()
			visited.add(cur_nid)

			if depth >= steps:
				continue 
			
			#print(cur_nid, len(graph[cur_nid]))

			for nei in graph[cur_nid]:
				if nei in visited:
					continue 

				queue.append((nei, depth + 1))

		return list(visited)

	def propagateByDistance(self, graph, nid, steps = 400):
		visited = set()
		queue = [(nid,0)]

		def distance(p1, p2):
			a = (p1[0] - p2[0])**2
			b = (p1[1] - p2[1])**2
			return np.sqrt(a + b)

			
		while len(queue) > 0:
			cur_nid, depth = queue.pop()
			visited.add(cur_nid)

			if depth >= steps:
				continue 
			
			#print(cur_nid, len(graph[cur_nid]))

			for nei in graph[cur_nid]:
				if nei in visited:
					continue 

				
				queue.append((nei, depth + distance(cur_nid, nei)))

		return list(visited)


	def match(self, nodes1, nodes2, thr = 8):
		idx = rtree.index.Index()
		for i in range(len(nodes1)):
			x,y = nodes1[i]
			idx.insert(i, (x-1,y-1,x+1,y+1))

		pairs = []

		m = thr

		for i in range(len(nodes2)):
			x,y = nodes2[i]

			candidates = list(idx.intersection((x-m, y-m, x+m, y+m)))
			for n in candidates:
				x2,y2 = nodes1[n]

				r = (x2-x)**2 + (y2-y)**2
				if r < thr*thr:
					pairs.append((i,n,r))

		pairs = sorted(pairs, key=lambda x: x[2])

		matched = 0 
		prop_set = set()
		gt_set = set()
		
		for pair in pairs:
			n1, n2, _ = pair
			if n1 in prop_set or n2 in gt_set:
				continue

			prop_set.add(n1)
			gt_set.add(n2)
			matched += 1
		
		precision = float(matched) / len(nodes1)
		recall = float(matched) / len(nodes2)

		return precision, recall


	def topoMetric(self, gt_graph, gt_directions, prop_graph, prop_directions, thr = 8, topo_range = 400,  mask = None, directional_match = True):
		# prop_graph = self.interpolateGraph(self.prop_graph)
		# gt_graph = self.interpolateGraph(self.gt_graph)

		prop_nodes = prop_directions.keys()
		gt_nodes = gt_directions.keys()

		#prop_nodes = self.getNodesFromGraph(self.prop_graph)
		#gt_nodes = self.getNodesFromGraph(self.gt_graph)
		if mask is not None:
			mask = scipy.ndimage.imread(mask)
			if len(np.shape(mask)) == 3:
				mask = mask[:,:,0]

		print(len(prop_nodes), len(gt_nodes))

		idx = rtree.index.Index()

		for i in range(len(gt_nodes)):
			x,y = gt_nodes[i]
			if mask is not None and mask[int(x), int(y)] < 127:
				continue

			idx.insert(i, (x-1,y-1,x+1,y+1))

		pairs = []

		m = thr
		
		for i in range(len(prop_nodes)):
			x,y = prop_nodes[i]
			dir_1 = prop_directions[(x,y)]

			if mask is not None and mask[int(x), int(y)] < 127:
				continue

			candidates = list(idx.intersection((x-m, y-m, x+m, y+m)))
			for n in candidates:
				x2,y2 = gt_nodes[n]

				r = (x2-x)**2 + (y2-y)**2

				dir_2 = gt_directions[gt_nodes[n]]

				if directional_match:
					cdot = (dir_2[0] * dir_1[0] + dir_2[1] * dir_1[1])
				else:
					cdot = abs(dir_2[0] * dir_1[0] + dir_2[1] * dir_1[1])

				if r < thr*thr and cdot > 0.5:

					pairs.append((i,n,r))

		pairs = sorted(pairs, key=lambda x: x[2])

		matched = 0 
		prop_set = set()
		gt_set = set()
		ps, rs = [], []
		for pair in pairs:
			n1, n2, _ = pair
			if n1 in prop_set or n2 in gt_set:
				continue

			prop_set.add(n1)
			gt_set.add(n2)

			# compute precision and recall

			if matched % 10 == 0:
				t0 = time()

				nodes1 = self.propagateByDistance(prop_graph, prop_nodes[n1], steps = topo_range)
				nodes2 = self.propagateByDistance(gt_graph, gt_nodes[n2], steps = topo_range)

				t2 = time()

				p,r = self.match(nodes1, nodes2, thr = thr)

				t1 = time()
				#print(t1-t2, t2-t0, len(nodes1), len(nodes2))

				if matched == 10:
					img_debug = np.zeros((4096, 4096, 3), dtype=np.uint8)
					for i in range(len(nodes1)):
						cv2.circle(img_debug, (int(nodes1[i][0]), int(nodes1[i][1])), 3, (255,0,0),1)
					
					for i in range(len(nodes2)):
						cv2.circle(img_debug, (int(nodes2[i][0]), int(nodes2[i][1])), 1, (0,0,255),-1)

					cv2.circle(img_debug, (int(prop_nodes[n1][0]),int(prop_nodes[n1][1])), 5, (0,255,0),1)
					cv2.circle(img_debug, (int(gt_nodes[n2][0]),int(gt_nodes[n2][1])), 2, (0,255,0),-1)

					cv2.imwrite("debug.png", img_debug)	
				#exit()


				ps.append(p)
				rs.append(r)

			matched += 1

			if matched % 10 == 0:
				sys.stdout.write("\rmatched %d precision:%.3f recall:%.3f %.3f "% (matched, np.mean(ps), np.mean(rs), t1-t0) )
				sys.stdout.flush()	
		
		print("matched", matched)

		print("geo precision", float(matched) / len(prop_nodes))
		print("geo recall", float(matched) / len(gt_nodes))

		print("topo precision", float(matched) / len(prop_nodes) * np.mean(ps))
		print("topo recall", float(matched) / len(gt_nodes) * np.mean(rs))
		

		return float(matched) / len(prop_nodes), float(matched) / len(gt_nodes), np.mean(ps), np.mean(rs)


if __name__ == "__main__":
	geo_precisions, geo_recalls = [], []
	topo_precisions, topo_recalls = [], []

	for tid in [0,5,6,11,12,17,18,22,25,28,31]:
		e = Evaluator()

		# graph_gt, direction_gt = e.loadGraphV2("../dataset_evaluation/way_%d.json" % tid,"../dataset_evaluation/link_%d.json" % tid )
		# graph_prop, direction_prop = e.loadGraphV2("../all_results_resnet34v3/%d/ways.json" % tid,"../all_results_resnet34v3/%d/links.json" % tid, transpose = True)

		graph_gt, direction_gt = e.loadGraphV2("../dataset_evaluation/way_%d.json" % tid, None )
		graph_prop, direction_prop = e.loadGraphV2("../all_results_resnet34v3/%d/ways.json" % tid, None, transpose = True)

		# graph_gt, direction_gt = e.loadGraphV2("samples/gt/way_0.json",None )
		# graph_prop, direction_prop = e.loadGraphV2("samples/prop/ways.json",None, transpose = True)


		# e.visGraph("gt.png", graph_gt, direction_gt)
		# e.visGraph("prop.png", graph_prop, direction_prop)

		geo_p, geo_r, topo_p, topo_r = e.topoMetric(graph_gt, direction_gt, graph_prop, direction_prop)

		geo_precisions.append(geo_p)
		geo_recalls.append(geo_r)

		topo_precisions.append(topo_p * geo_p)
		topo_recalls.append(topo_r * geo_r)

	print("geo precision", np.mean(geo_precisions))
	print("geo recall", np.mean(geo_recalls))

	print("topo precision", np.mean(topo_precisions))
	print("topo recall", np.mean(topo_recalls))

	ret = {}
	ret["geo precision"] = np.mean(geo_precisions)
	ret["geo recall"] = np.mean(geo_recalls)
	ret["topo precision"] = np.mean(topo_precisions)
	ret["topo recall"] = np.mean(topo_recalls)

	#json.dump(ret, open("lanegraphall.json", "w"), indent=2)
	json.dump(ret, open("lanegraphdirected.json", "w"), indent=2)


	# e.loadGTFromWays(sys.argv[1])
	# e.loadPropFromGraph(sys.argv[2])
	# e.topoMetric(mask = sys.argv[3])   
	# python eval.py ../dataset_evaluation/way_5.json ../all_results/5/graph.p ../dataset_evaluation/regionmask_5.jpg 
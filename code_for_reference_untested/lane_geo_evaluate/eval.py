import sys 
import json 
import pickle 
import numpy as np 
import rtree 
import cv2 
import scipy.ndimage

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
	def setGT(self, neighbors):
		self.gt_graph = neighbors

	def loadPropFromGraph(self,name):
		self.prop_graph = pickle.load(open(name, 'r'))

	def setProp(self, neighbors):
		self.prop_graph = neighbors
	
	def getNodesFromGraph(self, graph): # place nodes every 0.25 meter 
		nodes = set()
		exist = set()
		for nid, nei in graph.items():
			for nn in nei:
				if (nid, nn) in exist or (nn, nid) in exist:
					continue
				x1,y1 = nid 
				x2,y2 = nn 

				exist.add((nid, nn))

				L = int(np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)) // 2 + 1
				L = max(2,L)
				for i in range(L):
					a = float(i) / (L-1)
					x = x1*a + x2*(1-a)
					y = y1*a + y2*(1-a)

					if x < 0 or x >= 4096 or y < 0 or y >= 4096:
						continue

					if (x,y) not in nodes:
						nodes.add((x,y))

		return list(nodes)

	def interpolateGraph(self, graph):
		newgraph = {}
		exist = set()

		for nid, nei in graph.items():
			for nn in nei:
				if (nid, nn) in exist or (nn, nid) in exist:
					continue
				x1,y1 = nid 
				x2,y2 = nn 

				exist.add((nid, nn))

				L = int(np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)) // 2 + 1
				L = max(2,L)
				last_node = (x1,y1)
				for i in range(1,L):
					a = 1.0 - float(i) / (L-1)
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

					if nk1 not in newgraph:
						newgraph[nk1] = [nk2]
					elif nk2 not in newgraph[nk1]:
						newgraph[nk1].append(nk2)

					if nk2 not in newgraph:
						newgraph[nk2] = [nk1]
					elif nk1 not in newgraph[nk2]:
						newgraph[nk2].append(nk1)

					last_node = (x, y)

		return newgraph 

	def propagate(self, graph, nid, steps = 200):
		visited = set()
		queue = [(nid,0)]
		while len(queue) > 0:
			cur_nid, depth = queue.pop()
			visited.add(cur_nid)

			if depth >= steps:
				continue 
			
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

		

	def geoMetric(self, thr = 8, mask = None):
		prop_nodes = self.getNodesFromGraph(self.prop_graph)
		gt_nodes = self.getNodesFromGraph(self.gt_graph)
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

			if mask is not None and mask[int(x), int(y)] < 127:
				continue

			candidates = list(idx.intersection((x-m, y-m, x+m, y+m)))
			for n in candidates:
				x2,y2 = gt_nodes[n]

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
		
		print("matched", matched)

		print("precision", float(matched) / len(prop_nodes))
		print("recall", float(matched) / len(gt_nodes))
		
		img = np.zeros((4096, 4096, 3), dtype=np.uint8)

		for i in range(len(gt_nodes)):
			x,y = int(gt_nodes[i][0]), int(gt_nodes[i][1])
			if i in gt_set:
				color = (0,255,0)
			else:
				color = (255,0,0)

			cv2.circle(img, (x,y), 2, color,1)
		
		for i in range(len(prop_nodes)):
			x,y = int(prop_nodes[i][0]), int(prop_nodes[i][1])
			if i in prop_set:
				color = (0,255,0)
			else:
				color = (0,0,255)

			cv2.circle(img, (x,y), 2, color,-1)
		
		cv2.imwrite("geo.png", img)

		return matched, len(prop_nodes), len(gt_nodes)

	def topoMetric(self, thr = 8, r = 4*50,  mask = None, verbose = True):
		prop_graph = self.interpolateGraph(self.prop_graph)
		gt_graph = self.interpolateGraph(self.gt_graph)

		prop_nodes = prop_graph.keys()
		gt_nodes = gt_graph.keys()

		#prop_nodes = self.getNodesFromGraph(self.prop_graph)
		#gt_nodes = self.getNodesFromGraph(self.gt_graph)
		if mask is not None:
			mask = scipy.ndimage.imread(mask)
			if len(np.shape(mask)) == 3:
				mask = mask[:,:,0]

		if verbose:
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

			if mask is not None and mask[int(x), int(y)] < 127:
				continue

			candidates = list(idx.intersection((x-m, y-m, x+m, y+m)))
			for n in candidates:
				x2,y2 = gt_nodes[n]

				r = (x2-x)**2 + (y2-y)**2
				if r < thr*thr:
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
				nodes1 = self.propagateByDistance(prop_graph, prop_nodes[n1])
				nodes2 = self.propagateByDistance(gt_graph, gt_nodes[n2])

				p,r = self.match(nodes1, nodes2, thr = thr)
			
				ps.append(p)
				rs.append(r)

			matched += 1

			if matched % 10 == 0:
				sys.stdout.write("\rmatched %d precision:%.3f recall:%.3f "% (matched, np.mean(ps), np.mean(rs)) )
				sys.stdout.flush()	
		
		if verbose:
			print("matched", matched)

			print("geo precision", float(matched) / len(prop_nodes))
			print("geo recall", float(matched) / len(gt_nodes))

			print("topo precision", float(matched) / len(prop_nodes) * np.mean(ps))
			print("topo recall", float(matched) / len(gt_nodes) * np.mean(rs))
			

		return float(matched) / len(prop_nodes), float(matched) / len(gt_nodes), np.mean(ps), np.mean(rs)


if __name__ == "__main__":
	e = Evaluator()
	e.loadGTFromWays(sys.argv[1])
	e.loadPropFromGraph(sys.argv[2])
	e.topoMetric(mask = sys.argv[3])   
	# python eval.py ../dataset_evaluation/way_5.json ../all_results/5/graph.p ../dataset_evaluation/regionmask_5.jpg 
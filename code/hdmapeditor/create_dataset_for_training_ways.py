# 25.812813, -80.202372

import json 
import math 
from subprocess import Popen 

import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

from satellite import mapbox as md
from PIL import Image 
import numpy as np 
import scipy.misc
import json 
import cv2
import pickle
from roadstructure import LaneMap 



regions = json.load(open(sys.argv[1]))
inputfolder = sys.argv[2]
outputfolder = sys.argv[3]

counter = 0 
counter_out = 0
total_length = 0 
total_ways = 0
for region in regions:
	min_lat, min_lon = region["lat"], region["lon"]
	region_tag = region["tag"]
	regionsize = 4096
	stride = 4096
	tilesize = 4096
	res = 8

	blocks = [region["ilat"],region["ilon"]]
	folder = inputfolder

	for ilat in range(blocks[0]):
		for ilon in range(blocks[1]):
			subregion = [min_lat + ilat * stride / res / 111111.0, min_lon + ilon * stride / res / 111111.0 / math.cos(math.radians(min_lat))]
			subregion += [min_lat + (ilat * stride + tilesize) / res / 111111.0, min_lon + (ilon * stride + tilesize) / res / 111111.0 / math.cos(math.radians(min_lat))]

			#img = cv2.imread(folder + "/sat_%d.jpg" % (counter))
			try:
				labels = pickle.load(open(folder + "/sat_%d_label.p" % (counter), "rb"))
			except:
				break
			
			roadlabel, masklabel = labels 

			# find all ways 
			# - find nodes that belong to terminals or way intersections.
			# - search for a path between pairs of them.
			# - store them into a data structure for future use. 
			
			terminal_nodes = []
			for nid in roadlabel.nodes.keys():
				way_c = 0
				link_c = 0
				for nn in roadlabel.neighbors_all[nid]:
					if (nid, nn) in roadlabel.edgeType:
						edgetype = roadlabel.edgeType[(nid, nn)]
					else:
						edgetype = roadlabel.edgeType[(nn, nid)]

					if edgetype == "way":
						way_c += 1
					else:
						link_c += 1

				if way_c > 0 and link_c > 0:
					terminal_nodes.append(nid)
				elif way_c > 2 and link_c == 0:
					terminal_nodes.append(nid)
				elif way_c == 1 and link_c == 0:
					terminal_nodes.append(nid)
					
			wayset = set()
			ways = []
			for nid in terminal_nodes:
				queue = [[nid, []]]
				while len(queue) > 0:
					cur, curlist = queue.pop()
					#print(cur, curlist)
					curlist.append(cur)
					if cur in terminal_nodes and len(curlist) > 1 :
						ways.append(list(curlist))
						wayset.add((curlist[0], curlist[-1]))
						continue

					if len(curlist) > 1 and cur in curlist[:-1]:
						print("bug?")
						ways.append(list(curlist))
						wayset.add((curlist[0], curlist[-1]))
						continue
 

					for nn in roadlabel.neighbors[cur]:
						if roadlabel.nodeType[nn] == "way":
							if (cur, nn) in roadlabel.edgeType:
								edgetype = roadlabel.edgeType[(cur, nn)]
							else:
								edgetype = roadlabel.edgeType[(nn, cur)]
							if edgetype == "way":
								queue.append([nn, list(curlist)])

						#print(nn, curlist, cur in terminal_nodes, nn in terminal_nodes)

			def distance(a,b):
				return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

			newways = []
			for way in ways:
				node_list = [(roadlabel.nodes[nid][0], roadlabel.nodes[nid][1]) for nid in way]
				density = 50
				pd = [0]
				for i in range(len(node_list)-1):
					pd.append(pd[-1] + distance(node_list[i], node_list[i+1]))
				
				interpolate_N = int(pd[-1]/density)

				last_loc = node_list[0]
				nway = [last_loc]

				for i in range(interpolate_N):
					int_d = pd[-1]/(interpolate_N+1)*(i+1)
					for j in range(len(node_list)-1):
						if pd[j] <= int_d and pd[j+1] > int_d:
							a = (int_d-pd[j]) / (pd[j+1] - pd[j])

							loc = ((1-a) * node_list[j][0] + a * node_list[j+1][0], (1-a) * node_list[j][1] + a * node_list[j+1][1])
							nway.append((int(loc[0]), int(loc[1])) )
							last_loc = loc 
				nway.append(node_list[-1])
				newways.append(nway)
			
			ways = newways

			# nidmap = {}
			# for item in wayset:
			# 	n1,n2 = item 
			# 	if n1 not in nidmap:
			# 		nidmap[n1] = [n2]
			# 	else:
			# 		nidmap[n1].append(n2)

			# 	if n2 not in nidmap:
			# 		nidmap[n2] = [n1]
			# 	else:
			# 		nidmap[n2].append(n1)

			# for nid in terminal_nodes:
			# 	if nid not in nidmap:
			# 		nidmap[nid] = []
	
			print("number of ways", len(ways))
			#exit()
			#polygons = masklabel.findAllPolygons()
			# render masks, lanes, and normals (directions)
			
				

			for sr in [0, 1024, 2048]:
				for sc in [0, 1024, 2048]:
					mask = cv2.imread(outputfolder + "/regionmask_%d.jpg" % (counter_out))
					margin = 128

					localways = []

					neighbors = {}
					def addEdge(p1,p2):
						if p1 not in neighbors:
							neighbors[p1] = [p2]
						elif p2 not in neighbors[p1]:
							neighbors[p1].append(p2)

						if p2 not in neighbors:
							neighbors[p2] = [p1]
						elif p1 not in neighbors[p2]:
							neighbors[p2].append(p1)


					for way in ways:
						vertices = []
						outOfRange = False
						in_cc = 0
						out_cc = 0
						last_x = None 
						last_y = None 
						for loc in way:
							x = loc[0] - sc - margin
							y = loc[1] - sr - margin
							
								
							if x > 0 and x < 2048 and y > 0 and y < 2048 and mask[y,x,0] > 127:
								in_cc += 1
								vertices.append([x,y])
							else:
								out_cc += 1
								#outOfRange = True

							x = int(x)
							y = int(y)
							if last_x is not None:
								if (x > 0 and x < 2048 and y > 0 and y < 2048) or (last_x > 0 and last_x < 2048 and last_y > 0 and last_y < 2048):
									addEdge((y,x), (last_y, last_x))
							
							last_x = x 
							last_y = y 


						if len(vertices) >= 2:
							localways.append(vertices)
						# if in_cc >= out_cc * 2:
						# 	localways.append(vertices)
					
					# localnodes = {}

					# for nid in terminal_nodes:
					# 	x = roadlabel.nodes[nid][0] - sc - margin
					# 	y = roadlabel.nodes[nid][1] - sr - margin
					# 	if x > 0 and x < 4096 - 0 and y > 0 and y < 4096 - 0 and mask[y,x,0] > 127:
					# 		localnodes[nid] = [x,y]
					print(len(ways), " --> ", len(localways))
					json.dump(localways, open(outputfolder + "/way_%d.json" % (counter_out), "w"), indent=2)
					pickle.dump(neighbors, open(outputfolder + "/graph_%d.p" % (counter_out), "wb"), protocol=2)
					
					graph_vis = np.zeros((2048,2048), dtype=np.uint8)
					for n1, nei in neighbors.items():
						x1 = n1[1]
						y1 = n1[0]
						for nn in nei:
							x2 = nn[1]
							y2 = nn[0]
							cv2.line(graph_vis, (x1,y1), (x2,y2), (127), 2)
					for n1, nei in neighbors.items():
						x1 = n1[1]
						y1 = n1[0]
						cv2.circle(graph_vis, (x1,y1), 3, (255), -1)

					cv2.imwrite(outputfolder + "/graphvis_%d.jpg" % (counter_out), graph_vis)
						
						
					
					total_ways += len(localways)
					counter_out += 1
					print(counter_out)

				
			
			
			
			counter += 1
			
print(total_length, total_length / 8 / 1000.0)
print(total_ways)

# python3 create_dataset_for_training_ways.py dataset/regions.json dataset/ dataset_training/
# 25.812813, -80.202372

import json 
import math 
from subprocess import Popen 

import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

from satellite import mapbox as md
from osm import osm 
from PIL import Image 
import numpy as np 
import scipy.misc
import json 
import cv2
import pickle
from roadstructure import LaneMap 

def distance(p1, p2):
	a = p1[0] - p2[0]
	b = p1[1] - p2[1]
	return np.sqrt(a*a + b*b)

def waydistance_(way1, way2):
	d1 = 0
	for i in range(len(way1)):
		p0 = way1[i]
		min_d = 100000000
		for j in range(len(way2)-1):
			p1 = way2[j]
			p2 = way2[j+1]
			L = distance(p1,p2)
			L = L / 3
			for k in range(int(L)):
				a = float(k) / L 
				p3 = [p1[0] * a + p2[0] * (1-a), p1[1] * a + p2[1] * (1-a)]
				
				d = distance(p0, p3)
				min_d = min(min_d, d)

			if min_d < 8*10:
				break

		
		d1 = max(d1, min_d)
	return d1

def waydistance(way1, way2):
	return min(waydistance_(way1, way2), waydistance_(way2, way1))



regions = json.load(open(sys.argv[1]))
inputfolder = sys.argv[2]
outputfolder = sys.argv[3]

counter = 0 
counter_out = 0
total_length = 0 
c_road = 0
c_oneway_road = 0 

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

			# try:
			# 	labels = pickle.load(open(folder + "/sat_%d_label.p" % (counter), "rb"))
			# except:
			# 	break
			# roadlabel, masklabel = labels 

			#osmmap = osm.OSMLoader(subregion,noUnderground=True, includeServiceRoad=False)
			# TODO draw it 
			
			#polygons = masklabel.findAllPolygons()
			# render masks, lanes, and normals (directions)
			for sr in [0]:
				for sc in [0]:
					localways = json.load(open(outputfolder + "/way_%d.json" % (counter_out), "r"))
					
					distances = {}
					for i in range(len(localways)):
						#print(i, len(localways))
						for j in range(i,len(localways)):
							way1 = localways[i]
							way2 = localways[j]

							distances[(i,j)] = waydistance(way1, way2)
							distances[(j,i)] = distances[(i,j)]
							
					group_id = {}
					visited = set()
					d_thr = 8 * 10
					gid = 0 
					for i in range(len(localways)):
						if i in visited:
							continue

						queue = [i]
						
						while len(queue) > 0:
							cur = queue.pop()
							group_id[cur] = gid
							visited.add(cur)
							for j in range(len(localways)):
								if distances[(cur, j)] < d_thr and j not in visited:
									queue.append(j)

						gid += 1
					
					
					colors = []
					for r in range(32,256,32):
						for g in range(32,256,32):
							for b in range(32,256,32):
								colors.append((r,g,b))
					np.random.shuffle(colors)
					img = np.zeros((4096,4096,3), dtype=np.uint8)

					onewaygid = set()
					for gid_ in range(gid):
						directions = []
						for i in range(len(localways)):
							if group_id[i] == gid_:
								directions.append((localways[i][-1][0] - localways[i][0][0], localways[i][-1][1] - localways[i][0][1]))

						c_pos = 0
						c_neg = 0
						for j in range(len(directions)):
							for k in range(j+1, len(directions)):
								cdot = directions[j][0] * directions[k][0] + directions[j][1] * directions[k][1]
								if cdot >= 0:
									c_pos += 1
								else:
									c_neg += 1
						
						if c_neg == 0:
							onewaygid.add(gid_)

					print("find %d groups and %d one-way roads" % (gid, len(onewaygid)))
					c_road += gid
					c_oneway_road += len(onewaygid)
					
					for i in range(len(localways)):
						color = colors[group_id[i] % len(colors)]
						way = localways[i]

						if group_id[i] in onewaygid:
							width = 5
						else:
							width = 2
						for j in range(len(way)-1):
							if width == 5:
								cv2.line(img, (way[j][0], way[j][1]), (way[j+1][0], way[j+1][1]), (0,0,255), 10)

							cv2.line(img, (way[j][0], way[j][1]), (way[j+1][0], way[j+1][1]), color, width)

					group_id_list = []
					for i in range(len(localways)):
						gid = group_id[i]

						group_id_list.append([gid, gid in onewaygid])
					
					json.dump(group_id_list, open(outputfolder + "/group_%d.json" % (counter_out), "w"), indent=2)

					cv2.imwrite(outputfolder + "/group_%d.jpg" % (counter_out), img)
					
					
					
					
					
					counter_out += 1
					print(counter_out, c_road, c_oneway_road)
			counter += 1
			
print(total_length, total_length / 8 / 1000.0)



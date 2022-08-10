import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import sys 

from model_way import WayModel

from PIL import Image 
import numpy as np 
from subprocess import Popen 
import tensorflow as tf 
import math
import scipy.ndimage
import pickle
import cv2
import json
from segtograph.segtographfunc import segtograph

inputdirection = scipy.ndimage.imread(sys.argv[1])
inputgraph = pickle.load(open(sys.argv[2]))
outputfolder = sys.argv[3]

ways = []
visited = set()

for nid, nei in inputgraph.items():
    if len(nei) != 2:
        for next_node in nei:
            way = [nid]
            cur = next_node
            while True:
                way.append(cur)
                cur_nei = inputgraph[cur]
                if len(cur_nei) == 2:
                    if way[-2] == cur_nei[0]:
                        cur = cur_nei[1]
                    else:
                        cur = cur_nei[0]
                else:
                    break
            k1 = (way[0], way[-1])
            k2 = (way[-1], way[0])

            if k1 in visited or k2 in visited:
                pass
            else:
                ways.append(way)
            
            visited.add(k1)
            visited.add(k2)

# change directions
newways = []
for way in ways:
    cdot = 0
    for i in range(len(way)-1):
        r1,c1 = way[i][0],way[i][1]
        r2,c2 = way[i+1][0],way[i+1][1]
        l = math.sqrt((r1-r2)**2 + (c1-c2)**2)
        dr = (r2 - r1) / l
        dc = (c2 - c1) / l

        vr,vc = 0, 0
        for i in range(int(l)):
            a = i / float(l)
            r = int(r1 * a + r2 * (1-a))
            c = int(c1 * a + c2 * (1-a))

            vr += (inputdirection[r,c,1] - 127) / 127.0 
            vc += (inputdirection[r,c,2] - 127) / 127.0 
            
        l = math.sqrt((vr)**2 + (vc)**2)
        vr /= l
        vc /= l

        cdot += dr * vr + dc * vc 


    if cdot < 0:
        way.reverse()

json.dump(ways, open(outputfolder + "/ways.json", "w"), indent=2)

directions = np.zeros_like(inputdirection).astype(np.uint8) + 255
for way in ways:
	for i in range(len(way)-1):
		x1,y1 = way[i][1], way[i][0]
		x2,y2 = way[i+1][1], way[i+1][0]

		dx = x2-x1
		dy = y2-y1
		l = math.sqrt(dx*dx + dy*dy) + 0.00001
		dx /= l
		dy /= l

		color = (int(127 + 127*dx),int(127 + 127*dy),127)
		cv2.line(directions, (int(x1),int(y1)), (int(x2),int(y2)), color, 5)

cv2.imwrite(outputfolder + "/direction_sharp.png", directions)

            
            

        
         


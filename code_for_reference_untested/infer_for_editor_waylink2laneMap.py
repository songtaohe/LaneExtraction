# python3 
import os 
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

from hdmapeditor.roadstructure import LaneMap
import json 
import pickle 

ways = json.load(open(sys.argv[1]))
links = json.load(open(sys.argv[2]))
laneMap = LaneMap()

pos2nid = {}

for way in ways:
    for i in range(len(way)-1):
        x1,y1 = int(way[i][1]) + 128, int(way[i][0]) + 128
        x2,y2 = int(way[i+1][1]) + 128, int(way[i+1][0]) + 128

        if (x1,y1) not in pos2nid:
            pos2nid[(x1,y1)] = laneMap.addNode((x1,y1),False)
        nid1 = pos2nid[(x1,y1)]

        if (x2,y2) not in pos2nid:
            pos2nid[(x2,y2)] = laneMap.addNode((x2,y2),False)
        nid2 = pos2nid[(x2,y2)]

        laneMap.addEdge(nid1, nid2)

for link in links:
    for i in range(len(link)-1):
        x1,y1 = int(link[i][1]) + 128, int(link[i][0]) + 128
        x2,y2 = int(link[i+1][1]) + 128, int(link[i+1][0]) + 128

        if (x1,y1) not in pos2nid:
            pos2nid[(x1,y1)] = laneMap.addNode((x1,y1),False)
        nid1 = pos2nid[(x1,y1)]

        if (x2,y2) not in pos2nid:
            pos2nid[(x2,y2)] = laneMap.addNode((x2,y2),False)
        nid2 = pos2nid[(x2,y2)]

        laneMap.addEdge(nid1, nid2, "link")


pickle.dump([laneMap, LaneMap()], open(sys.argv[3], "wb"))

        
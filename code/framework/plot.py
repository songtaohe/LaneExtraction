import json
import sys 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 

log = json.load(open(sys.argv[1]))

if len(sys.argv) > 3:
    k = sys.argv[3]
else:
    k = "loss"

fig = plt.figure(figsize=(10,8), dpi = 100)

x = log[k][0][2:]
y = log[k][1][2:]

y_sorted = sorted(y)
y_min = y_sorted[0]
y_max = y_sorted[int(len(y) * 0.98)]

r = y_max - y_min 
y_max += r * 0.1
y_min -= r * 0.1

y_smooth = []
v = np.mean(y[0:30])
for i in range(len(y)):
    v = v * 0.99 + 0.01 * y[i]
    y_smooth.append(v)


plt.plot(x,y)
plt.plot(x,y_smooth)
plt.ylim([y_min, y_max])
plt.grid(True)
fig.tight_layout()

plt.savefig(sys.argv[2]) 

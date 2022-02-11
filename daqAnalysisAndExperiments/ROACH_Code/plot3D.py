from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
  
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
 
x = 3.0697
z = 2.4666
y = 3.684 

ax.plot3D(*zip([0, 0, 0], [x, 0, 0]), color="b")
ax.plot3D(*zip([0, 0, 0], [0, y, 0]), color="b")
ax.plot3D(*zip([0, 0, 0], [0, 0, z]), color="b")
ax.plot3D(*zip([x, 0, 0], [x, y, 0]), color="b")
ax.plot3D(*zip([x, 0, 0], [x, 0, z]), color="b")
ax.plot3D(*zip([x, 0, 0], [x, 0, z]), color="b")
ax.plot3D(*zip([0, y, 0], [x, y, 0]), color="b")
ax.plot3D(*zip([0, y, 0], [0, y, z]), color="b")
ax.plot3D(*zip([0, 0, z], [x, 0, z]), color="b")
ax.plot3D(*zip([0, 0, z], [0, y, z]), color="b")
ax.plot3D(*zip([x, y, 0], [x, y, z]), color="b")
ax.plot3D(*zip([x, 0, z], [x, y, z]), color="b")
ax.plot3D(*zip([0, y, z], [x, y, z]), color="b")

pos = []
pos.append([1.671, 1.646, 1.055])
pos.append([1.235, 2.012, 1.158])
pos.append([1.874, 0.620, 2.059])
pos.append([0.824, 1.936, 1.210])
pos.append([1.492, 1.955, 2.131])
pos.append([0.469, 1.588, 0.937])
pos.append([1.638, 0.355, 2.2667])
pos.append([0.791, 1.731, 2.426])
pos.append([1.689, 2.220, 3.0142])
pos.append([1.696, 1.229, 0.289])
pos.append([1.781, 1.954, 2.073])
pos.append([2.151, 0.804, 1.737])
pos.append([2.751, 2.260, 1.621])
pos.append([1.013, 1.213, 0.557])
pos.append([1.033, 1.736, 2.873])
pos.append([1.355, 1.064, 3.032])
pos.append([1.335, 1.354, 2.246])
pos.append([0.873, 0.244, 1.320])
pos.append([1.9867, 1.343, 2.252])
pos.append([2.528, 0.942, 2.864])
pos.append([2.6237, 0.5656, 0.311])
pos.append([0.790, 0.872, 0.355])
pos.append([1.960, 1.027, 0.789])
pos.append([0.446, 0.543, 3.237])

xThird = [0, 0, 0]
yThird = [0, 0, 0]
zThird = [0, 0, 0]

for val in pos:
	if val[0] < x/3:
		xThird[0] = xThird[0] + 1
	elif val[0] > 2*x/3:
		xThird[2] = xThird[2] + 1
	else:
		xThird[1] = xThird[1] + 1
	if val[1] < z/3:
		yThird[0] = yThird[0] + 1
	elif val[1] > 2*z/3:
		yThird[2] = yThird[2] + 1
	else:
		yThird[1] = yThird[1] + 1
	if val[2] < y/3:
		zThird[0] = zThird[0] + 1
	elif val[2] > 2*y/3:
		zThird[2] = zThird[2] + 1
	else:
		zThird[1] = zThird[1] + 1


print('X-DIST: ' + str(xThird))
print('Y-DIST: ' + str(yThird))
print('Z-DIST: ' + str(zThird))


ant = [0.805, 0.2456, 2.297]
term = [2.7137, 0.113, 0.399]

for val in pos:
	ax.scatter(val[0], val[2], val[1], color='r', s=5)

ax.scatter(ant[0], ant[2], ant[1], color='g', s=10)
ax.scatter(term[0], term[2], term[1], color='b', s=10)
plt.savefig('MovingLocations.png', bbox_inches='tight', dpi = 300)
plt.show()


#Vertices
# (0, 0, 0)
# (x, 0, 0)
# (0, y, 0)
# (0, 0, z)
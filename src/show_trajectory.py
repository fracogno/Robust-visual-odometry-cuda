import matplotlib.pyplot as plt
import numpy as np

plt.ion()

fig, ax = plt.subplots()

plot = ax.scatter([], [])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

while True:

	array = []
	with open("../data/trajectory_points.txt", "r") as ins:
		for line in ins:
		    	tmp = line.split()
			array.append([float(tmp[0]), float(tmp[1])])

	array = np.array(array)
	plot.set_offsets(array)	

	# update x and ylim to show all points:
	ax.set_xlim(array[:, 0].min()- 0.5, array[:,0].max() + 0.5)
	ax.set_ylim(array[:, 1].min() - 0.5, array[:, 1].max() + 0.5)
	
	# update the figure
	fig.canvas.draw()

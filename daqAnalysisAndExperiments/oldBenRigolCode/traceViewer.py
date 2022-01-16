import matplotlib.pyplot as plt
import numpy as np
import plottingClass
import sys

# Very simple plotter

# Filename is taken in as a command line argument
fileName = sys.argv[1]

# Plot formatting
plt.rc('font', family='serif', weight = 'bold')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

# Create a plotting object 
holder = plottingClass.plottingClass(color = 'blue')
holder.setData(fileName = fileName)
holder.removeFirstPoint()

# Multiply the data to have better units
holder.setData(xData = 'mult 1E-6', yData = 'mult 1E9')

# I will update this - the plotting class can handle the plotting
plt.plot(holder.x, holder.y, '-b', linewidth = 2)


# Label the axes
plt.ylabel(r'Amplitude (nV)', fontsize=24, labelpad = 15)
plt.xlabel('Frequency (MHz)', fontsize=24, labelpad = 20)

# Turn of x- and y-axis offsets
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)


# Tick parameters
plt.gca().tick_params(labelsize=16)

# Show the plot
plt.show()

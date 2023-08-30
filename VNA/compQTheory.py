import numpy as np
import matplotlib.pyplot as plt 


x = 3.05
y = 2.45
z = 3.67
volume = x*y*z
surfaceArea = (x*y + x*z + y*z)*2

freqs = np.linspace(50, 1000, 1000)
permeability = 800
conductivity = 1.4e6
depth = np.sqrt(2/(2*np.pi*freqs*1e6*4*np.pi*1e-7*permeability*conductivity))

qVals = 3*volume / (2*permeability*surfaceArea*depth)

plt.plot(freqs, qVals, color = (1, 1-0.5859375, 1-0.53125), linewidth = 3)
plt.xlabel('Frequency (MHz)', fontsize = 16)
plt.ylabel('Q', fontsize = 16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([50, 1000])
plt.ylim([100, 2000])
plt.tight_layout()
plt.savefig('TheoreticalQ.png', dpi = 150)
plt.show()
# -*- coding: utf-8 -*-
#%%
"""
Created on Tue Mar  2 11:04:06 2021

@author: phys-simulation
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.close('all')

#nate room
if 0:
    print("Nate dimensions")
    x = 3.048
    y = 2.438
    z = 3.657
else:
    #actual room
    print("measured dimensions Jan 2022")
    x = 3.0697
    y = 2.4566
    z = 3.6840


c = 2.99E8




def f(m, n, p, x, y, z):
    return 2.99E8/2 * np.sqrt((m/x)**2 + (n/y)**2 + (p/z)**2)/1E6

'''
#print("n = %i",n)
print("n = ",n, "\n m = ", m, "\n p =", p, \
      "\n f =",  round(f(n,m,p,x,y,z,c), 2), "MHz")
'''
    
maxMode = 9
sf = 40 #search frequency 
df = 150  #delta frequency

fp = sf + df
fm = sf - df

freqarr =[]
for m in range(0,maxMode):
    for n in range(0,maxMode):
        for p in range(0,maxMode):
            freq = f(m, n, p, x, y, z)
            freqarr.append(freq)
            if freq < fp and freq > fm:
                print("m =",m, "\nn =", n, "\np =", p, \
                      "\nf =",  round(freq, 2), "MHz")
                print()
'''
plt.figure()
plt.close('all')
#plt.xlim(0,300)
plt.hist(freqarr)
'''
# %%

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:04:06 2021

@author: phys-simulation
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.close('all')
'''
#actual room
x = 3.048
y = 2.438
z = 3.657
'''

#proposed room in room as of 10/1/21
x = 2.8368
y = 2.2798
z = 3.353


c = 2.99E8

#n = 5
#m = 2
#p = 0


def f(m, n, p, x, y, z):
    return 2.99E8/2 * np.sqrt((m/x)**2 + (n/y)**2 + (p/z)**2)/1E6

'''
#print("n = %i",n)
print("n = ",n, "\n m = ", m, "\n p =", p, \
      "\n f =",  round(f(n,m,p,x,y,z,c), 2), "MHz")
'''
    
maxMode = 7
sf = 150 #search frequency 
df = 150  #delta frequency

fp = sf + df
fm = sf - df

freqarr =[]
for m in range(1):
    for n in range(1,maxMode):
        for p in range(1,maxMode):
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
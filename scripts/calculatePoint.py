#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 18:10:01 2021

@author: dradmin
"""

#calculate points in polar to cartisian

import numpy as np

pi = np.pi


r = .4
dtheta = pi/2
dx = .1


thetaArr = np.arange(0, 2*pi, dtheta)
xArr = np.arange(-.6, .7, dx)

for theta in thetaArr:
    for x in xArr:
        print(x, ',', r*np.sin(theta), ',',r*np.cos(theta))

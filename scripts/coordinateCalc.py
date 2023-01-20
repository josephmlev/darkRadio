#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:52:55 2022

@author: dradmin
"""

x = 3.0697 #comsol width, real E(ast)
y = 2.4566 #comsel depth, real V(ert)
z = 3.6840 #comsol height, real N(orth)

def comsolToReal(xIn = -x/2, yIn = -y/2 , zIn = -z/2):
    print('North = ', z/2 - zIn)
    print('East = ', xIn + x/2)
    print('Vertical = ', yIn + y/2)
    

def realToComsol(nIn = -z/2, eIn = x/2, vIn = y/2):
    print('X =', eIn - x/2)
    print('Y =', vIn - y/2)
    print('Z =', nIn + z/2)

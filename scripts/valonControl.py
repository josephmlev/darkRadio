#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:28:17 2022

@author: dark-radio
"""

import pyvisa

rm = pyvisa.ResourceManager()


#Finding Valon resource name
unplug = input('make sure valon is unplugged, press enter to confirm')
l1 = list(rm.list_resources())
plugin = input('plug in valon, press enter to confirm')
l2 = list(rm.list_resources())

difference = list(set(l2) - set(l1))
valonName = difference[0]
print(valonName)
inst = rm.open_resource(valonName)
#----------------
#Valon Control
while True:
    source = input("Choose source (1 or 2): ")
    freq = input("Input desired frequence, add H,M,G for units, ex: 1.7G = 1.7 GHz: ")
    att = input("Input desire attenuation, The attenuator has a range of 0dB to 31.5dB in 0.5dB steps: ")
    print(inst.query("S " + source + "; f " + freq))
    print(inst.query("S " + source + "; att " + att))

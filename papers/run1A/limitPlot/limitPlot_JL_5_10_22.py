import os
import sys
import numpy as np
sys.path.append(os.getcwd())
print(os.getcwd())
from numpy import *
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from scipy.stats import norm
from PlotFuncs import col_alpha,CurvedText
from PlotFuncs import MySaveFig
#%matplotlib inline

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


def FigSetup(xlab=r'Dark photon mass, $m_X$ [eV]',ylab='Kinetic mixing, $\chi$',\
             chi_min = 1.0e-18,chi_max = 1.0e0,\
             m_min = 3e-18,m_max = 1e5,\
             lw=2.5,lfs=40,tfs=25,tickdir='out',\
             Grid=False,Shape='Rectangular',mathpazo=True,\
             TopAndRightTicks=False,FrequencyAxis=True,FrequencyLabels=True,UnitAxis=True,f_rescale=1,\
            tick_rotation = 20,width=20,height=10,upper_tickdir='out'):

    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)

    #if mathpazo:
     #   mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']

    if Shape=='Wide':
        fig = plt.figure(figsize=(16.5,5))
    elif Shape=='Rectangular':
        fig = plt.figure(figsize=(16.5,11))
    elif Shape=='Custom':
        fig = plt.figure(figsize=(width,height))

    ax = fig.add_subplot(111)

    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs)

    ax.tick_params(which='major',direction=tickdir,width=2.5,length=13,right=TopAndRightTicks,top=TopAndRightTicks,pad=7)
    ax.tick_params(which='minor',direction=tickdir,width=1,length=10,right=TopAndRightTicks,top=TopAndRightTicks)


    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([m_min,m_max])
    ax.set_ylim([chi_min,chi_max])

    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=50)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    if Shape=='Rectangular':
        plt.xticks(rotation=tick_rotation)

    if Grid:
        ax.grid(zorder=0)

    if FrequencyAxis:
        ax2 = ax.twiny()



        ax2.set_xscale('log')
        ax2.tick_params(which='major',direction=upper_tickdir,width=2.5,length=13,pad=7)
        ax2.tick_params(which='minor',direction=upper_tickdir,width=1,length=10)
        locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=50)
        locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
        ax2.xaxis.set_major_locator(locmaj)
        ax2.xaxis.set_minor_locator(locmin)
        ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        if FrequencyLabels:
            ax2.set_xticks([1e-3,1e0,1e3,1e6,1e9,1e12,1*241.8*1e12,1000*241.8*1e12])
            ax2.set_xticklabels(['mHz','Hz','kHz','MHz','GHz','THz','eV','keV'])
        ax2.set_xlim([m_min*241.8*1e12/f_rescale,m_max*241.8*1e12/f_rescale])

        plt.sca(ax)
    return fig,ax
    
    
import matplotlib.patheffects as pe
pek=[pe.Stroke(linewidth=7, foreground='k'), pe.Normal()]

    

    
def Haloscopes(ax,col=[0.75, 0.2, 0.2],fs=17,projection=True,text_on=True):
    y2 = ax.get_ylim()[1]
    zo = 0.3
    
    HAYSTAC_col = 'indianred'
    CAPP_col = 'crimson'
    QUAX_col = 'r'
    ADMX_col = 'firebrick'
    
    # ADMX
    costh = sqrt(0.019)
    B = 7.6
    dat = loadtxt("limit_data/AxionPhoton/ADMX.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=ADMX_col,zorder=0.1,lw=3)

    B = 6.8
    dat = loadtxt("limit_data/AxionPhoton/ADMX2018.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=ADMX_col,zorder=0.1)

    B = 7.6
    dat = loadtxt("limit_data/AxionPhoton/ADMX2019_1.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=ADMX_col,zorder=0.1)

    B = 7.6
    dat = loadtxt("limit_data/AxionPhoton/ADMX2019_2.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=ADMX_col,zorder=0.1)
    
    B = 7.6
    dat = loadtxt("limit_data/AxionPhoton/ADMX2021.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=ADMX_col,zorder=0.1)

#     B = 3.11
#     dat = loadtxt("limit_data/AxionPhoton/ADMX_Sidecar_AC.txt")
#     dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
#     plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=ADMX_col,facecolor=ADMX_col,zorder=0.1)

#     B = 5.0
#     dat = loadtxt("limit_data/AxionPhoton/ADMX_SLIC.txt")
#     dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
#     plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=ADMX_col,facecolor=ADMX_col,zorder=100)

    
    
    B = 9
    dat = loadtxt("limit_data/AxionPhoton/HAYSTAC_highres.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=HAYSTAC_col,zorder=0.1)
    dat = loadtxt("limit_data/AxionPhoton/HAYSTAC_2020_highres.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=HAYSTAC_col,zorder=0.1)

    
    # CAPP
    B = 7.3
    dat = loadtxt("limit_data/AxionPhoton/CAPP-1.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=CAPP_col,zorder=0.1)

    B = 7.8
    dat = loadtxt("limit_data/AxionPhoton/CAPP-2.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=CAPP_col,zorder=0.1)

    B = 7.9
    dat = loadtxt("limit_data/AxionPhoton/CAPP-3.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*costh*dat[:,0]))
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor='none',facecolor=CAPP_col,zorder=0.1)

    # CAPP-3 [KSVZ]
    dat_min = dat[argmin(dat[:,1]),:]
    dat_min[1] = dat_min[1]*costh/sqrt(0.2)
    plt.plot([dat_min[0],dat_min[0]],[1e-10,dat_min[1]],'-',color=CAPP_col,lw=1.5,zorder=0.1)


    B = 8.1
    costh = sqrt(0.03)
    dat = loadtxt("limit_data/AxionPhoton/QUAX.txt")
    dat[:,1] = 1e-9*dat[:,1]*(B/(1.444e-3*0.023*dat[:,0]))
    plt.fill_between([dat[0,0],dat[0,0]],[y2,dat[0,1]],y2=y2,color=QUAX_col,zorder=0.1)


    if text_on: 
        plt.text(1.4e-6,0.5e-14,r'ADMX',fontsize=fs,color=ADMX_col,rotation=90,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.8e-5,0.1e-13,r'CAPP',fontsize=fs-2,color=CAPP_col,rotation=90,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.19e-4,3e-15,r'HAYSTAC',fontsize=fs-5,color=HAYSTAC_col,rotation=90,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.55e-4,5e-12,r'QUAX',fontsize=fs-5,color=QUAX_col,rotation=-90,rotation_mode='anchor',ha='center',va='center')

    return
    
    
def StellarBounds(ax,fs=19,text_on=True):
    y2 = ax.get_ylim()[1]
    # Stellar physics constraints

    # Globular clusters 
    HB_col = [0.01, 0.75, 0.24]
    HB = loadtxt("limit_data/DarkPhoton/RG.txt")
    plt.plot(HB[:,0],HB[:,1],color='k',alpha=0.5,zorder=0.9,lw=2)
    plt.fill_between(HB[:,0],HB[:,1],y2=y2,edgecolor=None,facecolor=HB_col,zorder=0.9)
    
    # Globular clusters 
    HB_col = 'DarkGreen'
    HB = loadtxt("limit_data/DarkPhoton/HB.txt")
    plt.plot(HB[:,0],HB[:,1],color='k',alpha=0.5,zorder=0.95,lw=2)
    plt.fill_between(HB[:,0],HB[:,1],y2=y2,edgecolor=None,facecolor=HB_col,zorder=0.95)

    # Solar bound
    Solar_col = 'ForestGreen'
    Solar = loadtxt("limit_data/DarkPhoton/Solar.txt")
    plt.plot(Solar[:,0],Solar[:,1],color='k',alpha=0.5,zorder=1.02,lw=2)
    plt.fill_between(Solar[:,0],Solar[:,1],y2=y2,edgecolor=None,facecolor=Solar_col,zorder=1.02)

    Solar = loadtxt("limit_data/DarkPhoton/Solar-Global.txt")
    plt.plot(Solar[:,0],Solar[:,1]/Solar[:,0],color='k',alpha=0.5,zorder=1.021,lw=2)
    plt.fill_between(Solar[:,0],Solar[:,1]/Solar[:,0],y2=y2,edgecolor=None,facecolor=Solar_col,zorder=1.021)

    if text_on:
        plt.text(0.8e2*(1-0.01),1.5e-14*(1+0.05),r'Solar',fontsize=fs,color='k',rotation=-41,rotation_mode='anchor',ha='center',va='center')
        plt.text(1e3*(1-0.01),0.7e-14*(1+0.05),r'HB',fontsize=fs,color='k',rotation=-38,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.8e4*(1-0.01),0.7e-14*(1+0.05),r'RG',fontsize=fs,color='k',rotation=-37,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.8e2,1.5e-14,r'Solar',fontsize=fs,color='w',rotation=-41,rotation_mode='anchor',ha='center',va='center')
        plt.text(1e3,0.7e-14,r'HB',fontsize=fs,color='w',rotation=-38,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.8e4,0.7e-14,r'RG',fontsize=fs,color='w',rotation=-37,rotation_mode='anchor',ha='center',va='center')
    return

    
def Xenon(ax,col='crimson',fs=23,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/Xenon1T.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)

    plt.plot(1e3*dat[:,0],dat[:,1],color='k',alpha=0.5,zorder=0.5,lw=2)
    plt.fill_between(1e3*dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.5)
    
    dat = loadtxt("limit_data/DarkPhoton/Xenon1T_S1S2.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)

    plt.plot(dat[:,0],dat[:,1],color='k',alpha=0.5,zorder=0.5,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.5)
    
    
    dat = loadtxt("limit_data/DarkPhoton/XENON1T_SE.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=0.5,zorder=0.5,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.5)
    
        
    dat = loadtxt("limit_data/DarkPhoton/XENON1T_Solar_S2.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=0.5,zorder=0.5,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.5)
    
    if text_on: 
        plt.text(8e2,3e-17,r'XENON',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
    
    return





def DAMIC(ax,col='salmon',fs=21,text_on=True):
    m1,y1 = loadtxt("limit_data/DarkPhoton/DM_combined.txt",unpack=True)
    dat = loadtxt("limit_data/DarkPhoton/DAMIC.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)

    y2 = interp(dat[:,0],m1,y1)
    dat[0,1] = y2[0]
    dat[-1,1] = y2[-1]
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.001,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.001)
    if text_on: 
        plt.text(6e-1,1.3e-14,r'DAMIC',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.plot([5e0,1e1],[3e-14,6e-14],'-',lw=2.5,color=col)
    return


def FUNK(ax,col='red',fs=21,text_on=True):
    m1,y1 = loadtxt("limit_data/DarkPhoton/DM_combined.txt",unpack=True)
    dat = loadtxt("limit_data/DarkPhoton/FUNK.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)*sqrt(2/3/0.27)

    y2 = interp(dat[:,0],m1,y1)
    dat[0,1] = y2[0]/1.1
    dat[-1,1] = y2[-1]/1.1
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.3,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.3)
    if text_on: 
        plt.text(2.6e-1,1e-13,r'FUNK',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.plot([9e-1,3e0],[3e-13,1e-12],'-',lw=2.5,color=col)
    return

def SENSEI(ax,col='firebrick',fs=21,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/SENSEI.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)

    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1)
    if text_on: 
        plt.text(1.7e0,1e-15,r'SENSEI',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.plot([7e0,1e1],[3e-15,9e-15],'-',lw=2.5,color=col)
    return

def SuperCDMS(ax,col=[0.4,0,0],fs=18,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/SuperCDMS.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=0.5,zorder=0.6,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.6)
    
    if text_on: 
        plt.text(0.5e1,1.5e-16,r'SuperCDMS',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.plot([5e1,0.8e2],[3e-16,9e-16],'-',lw=2.5,color=col)
    return

def Nanowire(ax,col='pink',fs=22,text_on=True):
    m1,y1 = loadtxt("limit_data/DarkPhoton/DM_combined.txt",unpack=True)
    dat = loadtxt("limit_data/DarkPhoton/WSi_Nanowire.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)
    y2 = interp(dat[:,0],m1,y1)
    dat[0,1] = y2[0]/1.1
    dat[-1,1] = y2[-1]/1.1
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.3,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.3)
    if text_on: 
        plt.text(5e-4,1e-10,r'WSi Nanowire',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.plot([9e-3,3e-3],[3e-10,9e-10],'-',lw=2.5,color=col)
    return

def LAMPOST(ax,col='red',fs=15,text_on=True):
    m1,y1 = loadtxt("limit_data/DarkPhoton/DM_combined.txt",unpack=True)
    dat = loadtxt("limit_data/DarkPhoton/LAMPOST.txt")
    dat[:,1] = dat[:,1]*sqrt(0.4/0.45)*sqrt(2/3/0.27)

    y2 = interp(dat[:,0],m1,y1)
    dat[0,1] = y2[0]/1.1
    dat[-1,1] = y2[-1]/1.1
    plt.plot(dat[:,0],dat[:,1],color=col,alpha=1,zorder=0,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0)
    if text_on: 
        plt.text(0.3e-1,5e-13,r'LAMPOST',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.plot([3e-1,0.6e0],[6e-13,1e-12],'-',lw=1.5,color=col)
    return

def Tokyo(ax,col='darkred',fs=15,text_on=True):
    m1,y1 = loadtxt("limit_data/DarkPhoton/DM_combined.txt",unpack=True)
    dat = loadtxt("limit_data/DarkPhoton/Tokyo-Dish.txt")
    dat[:,1] = dat[:,1]*sqrt(2/3/0.6)
    y2 = interp(dat[:,0],m1,y1)
    dat[0,1] = y2[0]/1.1
    dat[-1,1] = y2[-1]/1.1
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.4,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.4)

    
    dat = loadtxt("limit_data/DarkPhoton/Tokyo-Knirck.txt")
    dat[:,1] = dat[:,1]*sqrt(1/3/0.175)
    plt.fill_between(dat[:,0],dat[:,1],y2=1e0,edgecolor='k',facecolor=col,zorder=1.09)
    
    dat = loadtxt("limit_data/DarkPhoton/Tokyo-Tomita.txt")
    plt.plot([dat[1,0],dat[1,0]],[dat[1,1],1e0],'-',color=col,lw=3,zorder=0.2)
    if text_on: 
        plt.text(2.3e-4,2.5e-10,r'Tokyo-3',fontsize=fs,color=col,rotation=-90,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.45e-3,3e-8,r'Tokyo-2',fontsize=fs-2,color='k',rotation=90,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.3e-1,4e-12,r'Tokyo-1',fontsize=fs+4,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.plot([3e-1,4e0],[5e-12,8e-12],'-',lw=2.5,color=col)
    return
    

def Jupiter(ax,col='Green',fs=15,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/Jupiter.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=2,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=2)
    if text_on: 
        plt.text(0.1e-14*(1-0.02),4.5e-1*(1+0.07),r'Jupiter',fontsize=fs,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.1e-14,4.5e-1,r'Jupiter',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')
    return

def Earth(ax,col='DarkGreen',fs=17,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/Earth.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.9,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.9)
    if text_on: 
        plt.text(0.4e-13*(1-0.01),2e-1*(1+0.05),r'Earth',fontsize=fs,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.4e-13,2e-1,r'Earth',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')
    return


def Crab(ax,col=[0.1,0.4,0.1],fs=17,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/Crab.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.09999,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.09999)
    
#     dat = loadtxt("limit_data/DarkPhoton/Crab_2.txt")
#     plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.9,lw=2)
#     plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.9)
    if text_on: 
        plt.text(0.5e-6*(1-0.02),3e-1*(1+0.07),r'Crab',fontsize=fs,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.5e-6,3e-1,r'Crab',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')
    
        plt.text(0.8e-6*(1-0.02),0.9e-1*(1+0.07),r'nebula',fontsize=fs,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.8e-6,0.9e-1,r'nebula',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')
    
    return


def SHUKET(ax,col='maroon',fs=13,text_on=True,edge_on=False,lw=0.8):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/SHUKET.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)*sqrt(1/3/0.038)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.2)
    if edge_on:
        plt.plot(dat[:,0],dat[:,1],'k-',lw=lw,zorder=0.2)
    if text_on: 
        plt.text(3.5e-5,0.13e-12,r'SHUKET',fontsize=fs,color=col,rotation=-90,rotation_mode='anchor',ha='center',va='center')
    return

def DarkEfield(ax,col='darkred',fs=17,text_on=True,edge_on=False,lw=0.8):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/DarkEfield.txt")
    dat[:,1] = dat[:,1]*sqrt(1.64/5) # convert from 5 sigma CL to 95%
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)*sqrt(1/3/0.129)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.2)
    if edge_on:
        plt.plot(dat[:,0],dat[:,1],'k-',lw=lw,zorder=0.2)
    if text_on: 
        plt.text(0.8e-7/1.2,0.2e-12,r'Dark',fontsize=fs,color=col,rotation=90,rotation_mode='anchor',ha='center',va='center')
        plt.text(2e-7/1.2,0.2e-12,r'E-field',fontsize=fs,color=col,rotation=90,rotation_mode='anchor',ha='center',va='center')
    return

def WISPDMX(ax,col='crimson',fs=12,text_on=True,edge_on=False,lw=0.8):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/WISPDMX.txt")
    dat[:,1] = dat[:,1]*sqrt(0.3/0.45)*sqrt(1/3/0.23)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.201)
    if edge_on:
        plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=0.201,lw=lw)

    if text_on: 
        plt.text(9e-7,4.1e-12/1.2,r'WISP',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(9e-7,1.8e-12/1.2,r'DMX',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')

    return

def SQuAD(ax,col=[0.7,0,0],fs=12,text_on=True,lw=0.5,point_on=False,ms=10):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/SQuAD.txt")
    dat[:,1] = dat[:,1]*sqrt(0.4/0.45)*sqrt(1/3/0.019)
    plt.plot([dat[0,0],dat[0,0]],[y2,dat[0,1]],lw=lw,color=col,alpha=1,zorder=0.2)
    if point_on:
        plt.plot(dat[0,0],dat[0,1],'o',mfc=col,mec='k',mew=lw+1,zorder=0.2,markersize=ms)
    if text_on: 
        plt.text(36e-6,0.25e-14,r'SQuAD',fontsize=fs,color=col,rotation=-90,rotation_mode='anchor',ha='center',va='center')
    return

def DMPathfinder(ax,col='pink',fs=13,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/DM-Pathfinder.txt")
    dat[:,1] = dat[:,1]*sqrt(1/0.075)
    plt.plot([dat[0,0],dat[0,0]],[y2,dat[0,1]],lw=2,color=col,alpha=1,zorder=0.6)
    if text_on: 
        plt.text(2.1e-9,0.5e-8/1.9,r'DM',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(2.1e-9,0.2e-8/1.9,r'Pathfinder',fontsize=fs,color=col,rotation=0,rotation_mode='anchor',ha='center',va='center')

    return

def DarkMatter(ax,Witte_col='royalblue',Caputo_col='dodgerblue',Arias_col='navy',fs=20,projection=True,text_on=True):
    y2 = ax.get_ylim()[1]
    zo = 0.3
    
    # Combined limits
    dat = loadtxt("limit_data/DarkPhoton/DM_combined.txt")
    plt.plot(dat[:,0],dat[:,1],'-',color='w',alpha=1,zorder=zo+0.1,lw=2.5,path_effects=pek)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor='lightgray',zorder=zo,alpha=1.0)
    plt.plot([1e-16,dat[0,0]],[dat[0,1],dat[0,1]],'--',color='w',alpha=1,zorder=zo+0.1,lw=2.5,path_effects=pek)
    plt.fill_between([1e-16,dat[0,0]],[dat[0,1],dat[0,1]],y2=y2,edgecolor=None,facecolor='lightgray',zorder=zo+0.1,alpha=1.0)
    plt.plot(dat[40:,0],dat[40:,1],'--',color='w',alpha=1,lw=2.5,zorder=1000,solid_capstyle='round')
    
    # Individual limits
    dat2 = loadtxt("limit_data/DarkPhoton/Cosmology_Witte_inhomogeneous.txt")
    dat4 = loadtxt("limit_data/DarkPhoton/Cosmology_Caputo_HeII.txt",delimiter=',')
    dat5 = loadtxt("limit_data/DarkPhoton/Cosmology_Arias.txt")
    
    plt.fill_between(dat2[:,0],dat2[:,1],y2=y2,edgecolor='k',facecolor=Witte_col,zorder=0.305,alpha=0.8)
    plt.fill_between(dat4[:,0],dat4[:,1],y2=y2,edgecolor='k',facecolor=Caputo_col,zorder=0.305,alpha=0.8)
    plt.fill_between(dat5[:,0],dat5[:,1],y2=y2,edgecolor='k',facecolor=Arias_col,zorder=0.306,alpha=1)

    if text_on: 
        plt.gcf().text(0.295,0.42-0.04,r'DPDM HeII',fontsize=15,color='w',ha='center')
        plt.gcf().text(0.295,0.4-0.04,r'Reionisation',fontsize=15,color='w',ha='center')
        plt.gcf().text(0.295,0.38-0.04,r'(Caputo et al.)',fontsize=13,color='w',ha='center')

        plt.gcf().text(0.365,0.37,r'DPDM',fontsize=17,color='w',ha='center')
        plt.gcf().text(0.365,0.35,r'(Witte et al.)',fontsize=13,color='w',ha='center')

        plt.gcf().text(0.49,0.48,r'DPDM',fontsize=18,color='w',ha='center')
        plt.gcf().text(0.49,0.46,r'(Arias et al.)',fontsize=16,color='w',ha='center')

    return
    
def COBEFIRAS(ax,col=[0.1,0.2,0.5],text_on=True):
    y2 = ax.get_ylim()[1]   
    dat3 = loadtxt("limit_data/DarkPhoton/COBEFIRAS.txt",delimiter=',')
    plt.fill_between(dat3[:,0],dat3[:,1],y2=y2,edgecolor='k',facecolor=col,zorder=0.5,alpha=1)
    if text_on: 
        plt.gcf().text(0.29,0.70,r'COBE/FIRAS',fontsize=22,color='w',ha='center')
        plt.gcf().text(0.29,0.67,r'$\gamma \rightarrow X$',fontsize=22,color='w',ha='center')
    return


def LSW(ax,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/SPring-8.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.1001,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.45, 0.05, 0.1],zorder=1.1001)
    
    dat = loadtxt("limit_data/DarkPhoton/ALPS.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.091,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.55, 0.0, 0.16],zorder=1.091)
    
    dat = loadtxt("limit_data/DarkPhoton/LSW_UWA.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.09,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.6, 0.0, 0.2],zorder=1.09)
    
    dat = loadtxt("limit_data/DarkPhoton/LSW_ADMX.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.089,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.65, 0.1, 0.24],zorder=1.089)
    
#     dat = loadtxt("limit_data/DarkPhoton/LSW_CERN.txt")
#     plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.089,lw=2)
#     plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.65, 0.15, 0.2],zorder=1.089)
    
    dat = loadtxt("limit_data/DarkPhoton/CROWS.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.08,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.7, 0.2, 0.2],zorder=1.08)
    
    if text_on: 
        plt.text(0.4e-6,0.15e-3,r'LSW-ADMX',fontsize=17,color='w',rotation=-58,rotation_mode='anchor',ha='center',va='center')
        plt.text(1e-5,5e-5,r'LSW-UWA',fontsize=14,color='w',rotation=-56,rotation_mode='anchor',ha='center',va='center')
        
        plt.text(0.55e0*(1-0.02),0.9e-4*(1+0.08),r'LSW-SPring-8',fontsize=13,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.55e0,0.9e-4,r'LSW-SPring-8',fontsize=13,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')

        
        plt.text(1.2e-4*(1-0.02),0.9e-5*(1+0.08),r'ALPS',fontsize=25,color='k',rotation=-56,rotation_mode='anchor',ha='center',va='center')
        plt.text(1.2e-4,0.9e-5,r'ALPS',fontsize=25,color='w',rotation=-56,rotation_mode='anchor',ha='center',va='center')

        plt.text(0.75e-7*(1-0.01),9.9e-5*(1+0.05),r'CROWS',fontsize=24,color='k',rotation=-56,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.75e-7,9.9e-5,r'CROWS',fontsize=24,color='w',rotation=-56,rotation_mode='anchor',ha='center',va='center')
    return

def Coulomb(ax,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/Cavendish.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.07,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.7,0,0],zorder=1.07)
    
    dat = loadtxt("limit_data/DarkPhoton/PlimptonLawton.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.071,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor='crimson',zorder=1.071)
    
    dat = loadtxt("limit_data/DarkPhoton/Spectroscopy.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.11,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.4, 0.0, 0.13],zorder=1.11)
    
    dat = loadtxt("limit_data/DarkPhoton/AFM.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.5,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=[0.4, 0.2, 0.2],zorder=1.5)
    if text_on: 
        plt.text(2.5e-10*(1-0.02),0.35e-1*(1+0.08),r'Plimpton-Lawton',fontsize=15,color='k',rotation=-38,rotation_mode='anchor',ha='center',va='center')
        plt.text(2.5e-10,0.35e-1,r'Plimpton-Lawton',fontsize=15,color='w',rotation=-38,rotation_mode='anchor',ha='center',va='center')
        
        plt.text(3e1*(1-0.02),3e-1*(1+0.08),r'AFM',fontsize=20,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(3e1,3e-1,r'AFM',fontsize=20,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')

        plt.text(0.5e-8*(1-0.02),4e-6*(1+0.08),r'Cavendish-Coulomb',fontsize=23,color='k',rotation=-38,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.5e-8,4e-6,r'Cavendish-Coulomb',fontsize=23,color='w',rotation=-38,rotation_mode='anchor',ha='center',va='center')
        
        plt.text(0.2e2*(1-0.01),1e-3*(1+0.08),r'Spectroscopy',fontsize=23,color='k',rotation=-34,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.2e2,1e-3,r'Spectroscopy',fontsize=23,color='w',rotation=-34,rotation_mode='anchor',ha='center',va='center')
    
    return

def NeutronStarCooling(ax,col='#004d00',fs=18,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/NeutronStarCooling.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.1001,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.1001)
    
    if text_on: 
        plt.text(0.9e4*(1-0.03),0.4e-6*(1+0.05),r'Neutron stars',fontsize=fs,color='k',rotation=-43,rotation_mode='anchor',ha='center',va='center')    
        plt.text(0.9e4,0.4e-6,r'Neutron stars',fontsize=fs,color='w',rotation=-43,rotation_mode='anchor',ha='center',va='center')    
    return

def CAST(ax,col='maroon',fs=27,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/CAST.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.1,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.1)
    if text_on: 
        plt.text(4e-3*(1-0.01),0.8e-6*(1+0.08),r'CAST',fontsize=fs,color='k',rotation=-59,rotation_mode='anchor',ha='center',va='center')
        plt.text(4e-3,0.8e-6,r'CAST',fontsize=fs,color='w',rotation=-59,rotation_mode='anchor',ha='center',va='center')
    return

def SHIPS(ax,col='indianred',fs=20,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/SHIPS.txt")
    dat[:,1] = dat[:,1]/dat[:,0]
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.09,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.09)
    if text_on: 
        plt.text(0.6e-1*(1-0.05),0.08e-8*(1+0.1),r'SHIPS',fontsize=fs,color='k',rotation=-32,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.6e-1,0.08e-8,r'SHIPS',fontsize=fs,color='w',rotation=-32,rotation_mode='anchor',ha='center',va='center')
    return

def TEXONO(ax,col=[0.5, 0.0, 0.13],fs=15,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/TEXONO.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1.101,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1.101)
    if text_on: 
        plt.text(0.25e2*(1-0.01),0.1e-4*(1+0.08),r'TEXONO',fontsize=fs,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.25e2,0.1e-4,r'TEXONO',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')
    return

def IGM(ax,col='seagreen',fs=18,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/IGM.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=0.5,zorder=0.49,lw=2)

    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.49)
    
    if text_on: 
        plt.text(4e-12*(1-0.05),0.03e-7*(1+0.07),r'IGM',fontsize=fs,color='k',rotation=-39,rotation_mode='anchor',ha='center',va='center')
        plt.text(4e-12,0.03e-7,r'IGM',fontsize=fs,color='w',rotation=-39,rotation_mode='anchor',ha='center',va='center')
        plt.gcf().text(0.233*(1-0.005),0.565*(1+0.003),r'DPDM heating',color='k',fontsize=23)
        plt.gcf().text(0.233,0.565,r'DPDM heating',color='w',fontsize=23)

    return

def LeoT(ax,col='mediumseagreen',fs=18,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/LeoT.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=0.48,zorder=0.3061,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.3061)
    
    if text_on: 
        plt.text(7e-13*(1-0.05),0.2e-9*(1+0.07),r'Leo T',fontsize=fs,color='k',rotation=-39,rotation_mode='anchor',ha='center',va='center')
        plt.text(7e-13,0.2e-9,r'Leo T',fontsize=fs,color='w',rotation=-39,rotation_mode='anchor',ha='center',va='center')
    return

def GasClouds(ax,col='#00cc66',fs=18,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/GasClouds.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=0.48,zorder=0.306,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=0.306)
    
    if text_on: 
        plt.text(0.86e-13*(1-0.07),1e-10*(1+0.07),r'Gas clouds',fontsize=fs,color='k',rotation=-38,rotation_mode='anchor',ha='center',va='center')
        plt.text(0.86e-13,1e-10,r'Gas clouds',fontsize=fs,color='w',rotation=-38,rotation_mode='anchor',ha='center',va='center')
    return

def SuperMAG(ax,col='#b5403e',fs=18,text_on=True):
    y2 = ax.get_ylim()[1]
    dat = loadtxt("limit_data/DarkPhoton/SuperMAG.txt")
    plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1,lw=2)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1)
    
    if text_on: 
        plt.text(1.5e-17*(1-0.05),1e-1*(1+0.05)/1.4,r'Super',fontsize=fs,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(1.5e-17,1e-1/1.4,r'Super',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(1.5e-17*(1-0.05),0.2e-1*(1+0.05)/1.4,r'MAG',fontsize=fs,color='k',rotation=0,rotation_mode='anchor',ha='center',va='center')
        plt.text(1.5e-17,0.2e-1/1.4,r'MAG',fontsize=fs,color='w',rotation=0,rotation_mode='anchor',ha='center',va='center')
    
    return




# The following code is taken from arXiv:2105.04565v3

fig,ax = FigSetup()

# DPDM
DarkMatter(ax)

# Axion haloscopes
Haloscopes(ax)

# # LSW/Helioscopes
LSW(ax)
CAST(ax)
SHIPS(ax)

# Tests of coulomb law
Coulomb(ax)

# # Reactor neutrinos
TEXONO(ax)

# # Geomagnetic field
SuperMAG(ax)

# # DPDM searches
Xenon(ax)
DAMIC(ax)
SENSEI(ax)
SuperCDMS(ax)
FUNK(ax)
LAMPOST(ax)
Tokyo(ax)
SHUKET(ax)
DarkEfield(ax)
WISPDMX(ax)
SQuAD(ax)
DMPathfinder(ax)

# # Astrophysical boundse
StellarBounds(ax)
COBEFIRAS(ax)
Jupiter(ax)
Earth(ax)
Crab(ax)
IGM(ax)
LeoT(ax)
GasClouds(ax)
NeutronStarCooling(ax)

# BHSR
plt.fill_between([6.5e-15,2.9e-11],[1e-18,1e-18],y2=1,color='gray',edgecolor='none',zorder=-100,alpha=0.25)
plt.gcf().text(0.304,0.176,r'Black hole',fontsize=23,ha='center',rotation=0,color='k',alpha=0.8)
plt.gcf().text(0.304,0.145,r'superradiance',fontsize=23,ha='center',rotation=0,color='k',alpha=0.8)


# Final label
plt.arrow(0.435, 0.375, 0, -0.055, transform=fig.transFigure,figure=fig,
  length_includes_head=True,lw=2.5,
  head_width=0.012, head_length=0.028, overhang=0.13,
  edgecolor='k',facecolor='w',clip_on=False,zorder=-1)

plt.text(4e-9,0.8e-14,r'Dark',fontsize=27,ha='center')
plt.text(4e-9,0.15e-14,r'photon',fontsize=27,ha='center')
plt.text(4e-9,0.02e-14,r'DM',fontsize=27,ha='center')


#massConversion = 6.626 * 10**-34 / (1.609 * 10**-19)
#massVal = np.median(dataFreqs) * massConversion

#plt.plot([massVal], [epsVal], 'ko', markersize = 15)
#plt.text(2*massVal, epsVal, 'HEPCAT',fontsize=16, weight = 'bold', ha='left')
plt.savefig('HEPCAT_Limits.pdf')
plt.show()
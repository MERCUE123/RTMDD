# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from ctypes import *
import pyhdf.SD as HDF
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import interpolate
import glob
# x=np.fromfile(r"C:\Users\1\Desktop\风云3D\moidsSRF\AquaSRF\rtcoef_eos_2_modis_srf_ch16.txt")

# # XXX = np.fromfile(')
VIIRS_SRF_Files = glob.glob('D:\VIIRS\VIIRS_SRF'+ '/*.txt')
VIIRS_CH = list(np.arange(0, 16) )
SRF_CH = list(np.arange(0, 16) )
for i in range(0, 16):
    VIIRS_CH[i] = np.loadtxt(VIIRS_SRF_Files[i],skiprows=4)
    VIIRS_CH[i][:,0] = 1/VIIRS_CH[i][:,0]*10000000
    VIIRS_CH[i][:,0] = VIIRS_CH[i][:,0][::-1]
    SRF_CH[i] = VIIRS_CH[i][:,1]
    SRF_CH[i] = SRF_CH[i][::-1]


for i in range(0,16):
    start = int(VIIRS_CH[i][0,0])
    end = int(VIIRS_CH[i][-1,0])
    print(start,end)

    ninter = int((end-start)/2.5)+1
    xNEW = np.linspace(start,end,ninter)
    fp1 = interpolate.UnivariateSpline(VIIRS_CH[i][:,0], SRF_CH[i],s=0)
    yNew = fp1(xNEW)
    np.savetxt(f'D:\VIIRS\VIIRS_SRF\InterSRF\{i+1}.txt',yNew ,newline='\r\n',delimiter = ',',fmt='%.5f')
    plt.plot(VIIRS_CH[i][:,0], SRF_CH[i],linestyle = 'solid',color = 'teal')
    plt.show()
# XXX = pd.read_csv(r'C:\Users\1\Desktop\1.csv')
# XXX = np.array(XXX)
# x = XXX[:,2]
# x = x[::-1]
# y = XXX[:,3]
# y = y[::-1]

# s = 615
# e = 680
# n1 = (e-s)/2.5
# n = int(n1)
# xNEW = np.linspace(s,e,n+1)
# fp1 = interpolate.UnivariateSpline(x, y,s=0)

# Ynew = fp1(xNEW)

# np.savetxt(r'C:\Users\1\Desktop\1.txt',Ynew ,delimiter=',',fmt='%.5f')
# a=2
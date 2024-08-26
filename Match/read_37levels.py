# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:27:03 2021

@author: 90627
"""
#三个37层重采样
from netCDF4 import Dataset
import numpy as np
import h5py
from ctypes import *  #在python中调用C的库函数
import datetime as dt
import os.path   #os.path 模块主要用于获取文件的属性
import struct

sPath = 'D:/ERA5/202001/'
VAR = ['RH','Geopotential', 'Temperature']
KEY = ['r','z', 't']
nKeys = 3

sYear = '2020'
sMonth = '01'
nDays = 2
nHours = 24
nLevels = 37

nERA5_Var1 = 1
nERA5_Var37 = 3
nERA5_Levels = 37

nRow1 = 721
nCol1 = 1440
LON = np.fromfile('D:/ERA5/past/ERA5_lon.dat', dtype=np.float32)
LON.shape = nRow1, nCol1
LAT = np.fromfile('D:/ERA5/past/ERA5_lat.dat', dtype=np.float32)
LAT.shape = nRow1, nCol1

nRow2 = 180
nCol2 = 360
lon0 = 0.375
lat0 = 89.625
dxy = 1.0
nMinimumCount = 16
OUT = np.zeros((nRow2, nCol2), dtype=np.float32)
maxRH = np.zeros((nRow2, nCol2), dtype=np.float32)  #增加的一层最大相对湿度 

RSLib = cdll.LoadLibrary('D:/FY4A/202001/RSLib.so')
RSLib.Granule2Grid.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                               np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                               np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                               c_int,\
                               c_int,\
                               c_float,\
                               np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                               c_float,\
                               c_float,\
                               c_float,\
                               c_int,\
                               c_int,\
                               c_int]                 

for iKey in range(nKeys):
    sVar = VAR[iKey]
    sKey = KEY[iKey]
    iDay = 1
    while iDay <= nDays:
        if iDay < 10:
            sDay = '0' + str(iDay)
        else:
            sDay = str(iDay)
    
        sInFile = sPath + sYear + sMonth + sDay + '_' + sVar + '.nc'
        nc = Dataset(sInFile)
        print(sInFile)
        
        for iHour in range(nHours):
            if iHour < 10:
                sHour = '0' + str(iHour)
            else:
                sHour = str(iHour)
                
            sOutFile = sPath + sYear + sMonth + sDay + sHour + '_1X1_' + sVar + '.dat'
            fpout = open(sOutFile, 'wb')
            
            if sVar == 'RH':
                for i in range(nRow2):
                    for j in range(nCol2):
                    #0.25*4=1
                        maxRH[i,j]=nc['r'][iHour,:,4*i:4*(i+1), 4*j:4*(j+1)].max()
                fpout.write(maxRH)
                
                for level in range(nERA5_Levels):
                    IN = nc[sKey][iHour,level,:,:].astype(np.float32)
                    bSuccess = RSLib.Granule2Grid(IN, LON, LAT, nRow1, nCol1, 0, OUT, lon0, lat0, dxy, nRow2, nCol2, nMinimumCount)
                    fpout.write(OUT)
                fpout.close()
                
            
            else:
                for level in range(nERA5_Levels):
                    IN = nc[sKey][iHour,level,:,:].astype(np.float32)
                    bSuccess = RSLib.Granule2Grid(IN, LON, LAT, nRow1, nCol1, 0, OUT, lon0, lat0, dxy, nRow2, nCol2, nMinimumCount)
                    fpout.write(OUT)
                fpout.close()
        nc.close()
        iDay += 1    
        print('done')
    
print('finished!')
            
            
            
            
            
            
            
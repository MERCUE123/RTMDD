# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import sys
from ctypes import *
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
import datetime
import pandas as pd
from scipy import interpolate
nRow2 = 721
nCol2 = 1440
dxy = 0.25
suffix = '_0.25X0.25.dat'
fillvalue1 = -1
fillvalue2 = 0

def RSLibImport(path):
    RSLib = cdll.LoadLibrary(path)
    RSLib.Grid2Granule.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   c_float, \
                                   c_float, \
                                   c_float, \
                                   c_int, \
                                   c_int, \
                                   c_float, \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   c_int, \
                                   c_int, ]

    return RSLib

if __name__ == '__main__':
    sListFile = r'C:\Users\1\Desktop\py\Match\listAero.dat'
    fp = open(sListFile, 'r')
    filelist = [x.rstrip() for x in fp]
    fp.close()
    for k, sHdf in enumerate(filelist):
        nc = Dataset(filelist[k],'r')
        long= nc.variables['longitude'][:]
        lon = np.array(long)
        lati= nc.variables['latitude'][:]
        lat = np.array(lati)
        print(long[0],long[-1],lati[0],lati[-1])
        AOD550 = nc.variables['AOD550_mean'][:].data
        AOD670 = nc.variables['AOD670_mean'][:].data
        AOD870 = nc.variables['AOD670_mean'][:].data
        AOD1600 = nc.variables['AOD670_mean'][:].data
        AOD550 = np.where(AOD550[:,:] == -999,0,AOD550[:,:])
        AOD670 = np.where(AOD670[:,:] == -999,0,AOD670[:,:])
        AOD870 = np.where(AOD870[:,:] == -999,0,AOD870[:,:])
        AOD1600 = np.where(AOD1600[:,:] == -999,0,AOD1600[:,:])

        AOD550ravel = AOD550
        AOD670ravel = AOD670
        AOD870ravel = AOD870
        AOD1600ravel = AOD1600

        AOD550ravel.shape = 64800,1
        AOD670ravel.shape = 64800,1
        AOD870ravel.shape = 64800,1
        AOD1600ravel.shape = 64800,1


        # Rect双变量插值
        # AOD550interRect2 = interpolate.RectBivariateSpline(lat,lon,AOD550, kx=2,ky=2)
        # AOD550interRect1 = interpolate.RectBivariateSpline(lat,lon,AOD550, kx=1,ky=1)
        # xx = np.linspace(lati.min(),lati.max(),nRow2)
        # yy = np.linspace(long.min(),long.max(),nCol2)
        # AOD550interRect1 = AOD550interRect1(xx,yy)

        #这个数据太大跑不了
        # xx = np.linspace(-90,90,nRow2)
        # yy = np.linspace(-180,179.75,nCol2)
        # xx,yy= np.meshgrid(yv,xv,indexing = 'xy')
        # 原数据的经纬度
        xv = np.linspace(lat.max(),lat.min(),len(lat))
        yv  = np.linspace(lon.min()+180,lon.max()+180,len(lon))

        xv,yv= np.meshgrid(yv,xv,indexing = 'xy')



        # 采样点 mgrid左闭右开
        xx, yy = np.mgrid[
                90:-90:721*1j,
                0:359.75:1440 * 1j]


        xv = np.ravel(xv)
        yv = np.ravel(yv)

        all = np.transpose(np.vstack((xv,yv)))

        ## griddata插值 all = point(x,y)  AOD550 = data  (yy,xx) = sample point


        AOD550inter = interpolate.griddata(all,AOD550ravel,(yy,xx),fill_value=0)
        AOD670inter = interpolate.griddata(all,AOD670ravel,(yy,xx),fill_value=0)
        AOD870inter = interpolate.griddata(all,AOD870ravel,(yy,xx),fill_value=0)
        AOD1600inter = interpolate.griddata(all,AOD1600ravel,(yy,xx),fill_value=0)

        AOD550inter.shape = 721,1440
        AOD670inter.shape = 721,1440
        AOD670.shape = 180, 360
        AOD870inter.shape = 721,1440
        AOD1600inter.shape = 721,1440

        sOutFile = filelist[k].replace('.nc', suffix)
        ##
        fp = open(sOutFile, 'wb')
        fp.write(AOD550inter.astype(np.float32))
        fp.write(AOD670inter.astype(np.float32))
        fp.write(AOD870inter.astype(np.float32))
        fp.write(AOD1600inter.astype(np.float32))
        fp.close()
        # #画图
        # plt.subplot(1,2,1)
        # plt.imshow(AOD670inter)
        # plt.subplot(1, 2, 2)
        # plt.imshow( AOD670)
        # plt.show()
        sHdrFile = sOutFile.replace('.dat', '.hdr')
        fp = open(sHdrFile, 'w')
        fp.write('ENVI\ndescription = {\n  File Imported into ENVI.}\n')
        fp.write('samples = {0:d}\nlines   = {1:d}\nbands   = {2:d}\n'.format(nCol2, nRow2, 4))
        fp.write(
            'header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\ndata ignore value = -1\n'
            'sensor type = Unknown\nbyte order = 0\nwavelength units = Unknown\n')

        fp.write(
            'band names = {AOD550,AOD670,AOD870,AOD1600}')  # FY-3E MERSI
        # fp.write('band names = {CH24, CH25, VZA, VAA, RTime}') # FY-3D MERSI
        fp.close()
        print(sHdf + ' Finished!\n')





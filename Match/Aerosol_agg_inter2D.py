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
import os
from tqdm import tqdm
nRow2 = 721
nCol2 = 1440
dxy = 0.25
suffix = '_0.25X0.25.dat'
sOutPath = r'D:\ERA5\Aero\SampleAeroGrid2'

if __name__ == '__main__':
    sListFile = r'C:\Users\1\Desktop\py\Match\listAero.dat'
    fp = open(sListFile, 'r')
    filelist = [x.rstrip() for x in fp]
    fp.close()
    for k, sHdf in tqdm(enumerate(filelist)):
        nc = Dataset(filelist[k],'r')
        long= nc.variables['longitude'][:]
        lon = np.array(long)
        lati= nc.variables['latitude'][:]
        lat = np.array(lati)
        # aero原数据经度从-179.5-179.5  维度从-89.5-89.5
        print(long[0],long[-1],lati[0],lati[-1])

        # 读取原数据
        AOD550 = nc.variables['AOD550_mean'][:].data
        AOD670 = nc.variables['AOD670_mean'][:].data
        AOD870 = nc.variables['AOD870_mean'][:].data
        AOD1600 = nc.variables['AOD1600_mean'][:].data
        AOD550 = np.where(AOD550[:,:] == -999,0,AOD550[:,:])
        AOD670 = np.where(AOD670[:,:] == -999,0,AOD670[:,:])
        AOD870 = np.where(AOD870[:,:] == -999,0,AOD870[:,:])

        AOD1600 = np.where(AOD1600[:,:] == -999,0,AOD1600[:,:])
        # aero原数据的经纬度
        nRow1,nCol1 = AOD550.shape
        LON1 = np.zeros([nCol1])
        LAT1 = np.zeros([nRow1])
        for i in np.arange(nCol1) :
            LON1[i] = -179.5 + i
        for i in np.arange(nRow1):
            LAT1[i] = -89.5 + i

        
        q = 1 # 插值阶数
        f550 = interpolate.RectBivariateSpline(LAT1,LON1 ,AOD550,kx=q,ky=q,s=0)
        f670 = interpolate.RectBivariateSpline(LAT1,LON1,  AOD670,kx=q,ky=q,s=0) 
        f870 = interpolate.RectBivariateSpline(LAT1,LON1,  AOD870,kx=q,ky=q,s=0)
        f1600 = interpolate.RectBivariateSpline( LAT1,LON1, AOD1600,kx=q,ky=q,s=0)

        # 需要插值的经纬度
        LON2 = np.arange(-180, 180, 0.25)
        LAT2 = np.arange(-90, 90.25, 0.25)

        # 计算插值结果
        AOD550new = f550(LAT2,LON2 )
        AOD670new = f670(LAT2,LON2 )
        AOD870new = f870(LAT2,LON2 )
        AOD1600new = f1600(LAT2,LON2 )

        #  画图专用，运行注释掉
        # plt.subplot(1,2,1)
        # plt.imshow(AOD550)
        # plt.subplot(1, 2, 2)
        # plt.imshow( AOD550new)
        # plt.show()



        #  抽样
        for i in np.arange(721):
            
            for j in np.arange(1440):
                
                if AOD550new[i,j] <= 0.0001 :
                    continue 
                ib = int(i/4)
                jb = int(j/4)
                if AOD550[ib,jb] <= 0.0001 :
                    continue
                plt.title('AOD')
                
                # plt.subplot(2,2,1)
                AOD = np.array([550,670,870,1600])
                # AODorigin = 
                A550 = AOD550new[i,j]
                MAX = np.max(AOD670new)
                # plt.plot(AOD[0],AOD550new[i,j],'b*',AOD[1],AOD670new[i,j],'b*',AOD[2],AOD870new[i,j],'b*',AOD[3],AOD1600new[i,j],'b*')

                B550 = AOD550[ib,jb]
                # plt.plot(AOD[0],AOD550[ib,jb],'ro',AOD[1],AOD670[ib,jb],'ro',AOD[2],AOD870[ib,jb],'ro',AOD[3],AOD1600[ib,jb],'ro')
                # plt.show()


        AOD550new = np.ascontiguousarray(AOD550new,dtype=np.float32)
        AOD670new = np.ascontiguousarray(AOD670new,dtype=np.float32)
        AOD870new = np.ascontiguousarray(AOD870new,dtype=np.float32)
        AOD1600new = np.ascontiguousarray(AOD1600new,dtype=np.float32)

        sOutFile = os.path.basename(filelist[k].replace('.nc', suffix))
        sOutFile = os.path.join(sOutPath ,sOutFile)
        fp = open(sOutFile, 'wb')

        fp.write(AOD550new)
        fp.write(AOD670new)
        fp.write(AOD870new)
        fp.write(AOD1600new)
        fp.close()



        # Rect双变量插值
        # AOD550interRect2 = interpolate.RectBivariateSpline(lat,lon,AOD550, kx=2,ky=2)
        # AOD550interRect1 = interpolate.RectBivariateSpline(lat,lon,AOD550, kx=1,ky=1)
        # xx = np.linspace(lati.min(),lati.max(),nRow2)
        # yy = np.linspace(long.min(),long.max(),nCol2)
        # AOD550interRect1 = AOD550interRect1(xx,yy)

        #这个数据太慢跑不了
        # xx = np.linspace(-90,90,nRow2)
        # yy = np.linspace(-180,179.75,nCol2)
        # xx,yy= np.meshgrid(yv,xv,indexing = 'xy')
        # 原数据的经纬度

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
            'band names = {AOD550,AOD670,AOD870,AOD1600}')  # 
        # fp.write('band names = {CH24, CH25, VZA, VAA, RTime}') # FY-3D MERSI
        fp.close()
        print(sHdf + ' Finished!\n')





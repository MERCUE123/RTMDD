import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import glob
import re
import matplotlib.pyplot as plt

FY3D_SRF_Path   = r'C:\Users\1\Desktop\casestudy\data\MERSI\MERSI_SRF'
MODIS_SRF_Path  = r'C:/Users/1/Desktop/casestudy/data/MODIS\AquaSRF'

MODIS_SRF_Files = glob.glob(MODIS_SRF_Path + '/*.txt')
FY3D_SRF_Files  = glob.glob(FY3D_SRF_Path + '/*.txt')

# 正则给通道的SRF排序
# \d ：匹配任意数字  + ：匹配前面的字符一次或多次
# re.search()方法扫描整个字符串，并返回第一个成功的匹配。如果匹配失败，则返回None。
# re.match()方法要求必须从字符串的开头进行匹配，如果字符串的开头不匹配，整个匹配就失败了；
# a = lambda x:re.search(r'ch\d+', x).group()
# b = a(MODIS_SRF_Files[0])


MODIS_CH = list(np.arange(0, 19) )
for i in range(0, 19):
    MODIS_CH[i] = np.loadtxt(MODIS_SRF_Files[i],skiprows=4)
    MODIS_CH[i][:,0] = 1/MODIS_CH[i][:,0]*10000000

FY3D_CH = list(np.arange(0, 20) )
for i in range(0, 20):
    FY3D_CH[i] = np.loadtxt(FY3D_SRF_Files[i])

# 通道匹配
# MERSI = list(np.array([0.471,0.554,0.653,0.868,1.381,1.645,2.125,0.411,0.444,0.490,
#                 0.556,0.670,0.709,0.746,0.865,0.905,0.936,0.940,1.030])) 
# Aqua = list(np.array([0.645,0.856,0.466,0.554,1.241,1.628,2.113,0.412,0.442,0.487,
#                 0.530,0.547,0.666,0.678,0.747,0.867,0.904,0.936,0.935]))
 

# i for 3D and j for MODIS
MatchBand = [(1,3),(2,4),(3,1),(4,2),(5,5),(6,6),(7,7),(8,8),(9,9),
             (10,10),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),
             (17,18),(18,19),(19,19)]


# 画SRF
# 1-10 Band
for i,j in MatchBand:
    # 1通道是列表第0个，故减去1
    i = i-1
    j = j-1
    if i < 10:
        plt.subplot(2,5,i+1)
        plt.plot(FY3D_CH[i][:,0],FY3D_CH[i][:,1],linestyle = 'solid',color = 'teal')
        plt.plot(MODIS_CH[j][:,0],MODIS_CH[j][:,1],linestyle = 'solid',color = 'tomato')
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('SRF')
        # plt.legend(['MERISI-II','MODIS'])
        plt.title('MERSI-II%d' % (i+1)+ ' & '+'MODIS%d' % (j+1))
plt.tight_layout(h_pad=0.1,w_pad=0)
plt.show()

# 10-19 Band
for i,j in MatchBand:
    # 1通道是列表第0个，故减去1
    i = i-1
    j = j-1
    if i >= 10:
        plt.subplot(2,5,i-9)
        plt.plot(FY3D_CH[i][:,0],FY3D_CH[i][:,1],linestyle = 'solid',color = 'teal')
        plt.plot(MODIS_CH[j][:,0],MODIS_CH[j][:,1],linestyle = 'solid',color = 'tomato')
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('SRF')
        # plt.legend(['MERISI-II','MODIS'])
        plt.title('MERSI-II%d' % (i+1)+' & '+'MODIS%d' % (j+1))
plt.tight_layout(h_pad=0.1,w_pad=0)
plt.show()

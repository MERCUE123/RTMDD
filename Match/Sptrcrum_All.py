import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import glob
import re
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

MERSI_SRF_Path   = r'C:/Users/1/Desktop/casestudy/data/MERSI\MERSI_SRF'
VIIRS_SRF_Path  = r'C:/Users/1/Desktop/casestudy/data/VIIRS\VIIRS_SRF'
MODIS_SRF_Path  = r'C:/Users/1/Desktop/casestudy/data/MODIS\AquaSRF'

MODIS_SRF_Files = glob.glob(MODIS_SRF_Path + '/*.txt')
VIIRS_SRF_Files = glob.glob(VIIRS_SRF_Path + '/*.txt')
MERSI_SRF_Files  = glob.glob(MERSI_SRF_Path + '/*.txt')

# 正则给通道的SRF排序
# \d ：匹配任意数字  + ：匹配前面的字符一次或多次
# re.search()方法扫描整个字符串，并返回第一个成功的匹配。如果匹配失败，则返回None。
# re.match()方法要求必须从字符串的开头进行匹配，如果字符串的开头不匹配，整个匹配就失败了；
# a = lambda x:re.search(r'ch\d+', x).group()
# b = a(VIIRS_SRF_Files[0])


VIIRS_CH = list(np.arange(0, len(VIIRS_SRF_Files)) )
for i in range(0, len(VIIRS_CH)):
    VIIRS_CH[i] = np.loadtxt(VIIRS_SRF_Files[i],skiprows=4)
    VIIRS_CH[i][:,0] = 1/VIIRS_CH[i][:,0]*10000000

MERSI_CH = list(np.arange(0, 20) )
for i in range(0, 19):
    MERSI_CH[i] = np.loadtxt(MERSI_SRF_Files[i])
    
MODIS_CH = list(np.arange(0, 19) )
for i in range(0, 19):
    MODIS_CH[i] = np.loadtxt(MODIS_SRF_Files[i],skiprows=4)
    MODIS_CH[i][:,0] = 1/MODIS_CH[i][:,0]*10000000

# MERSI = list(np.array([0.471,0.554,0.653,0.868,1.381,1.645,2.125,0.411,0.444,0.490,
#                 0.556,0.670,0.709,0.746,0.865,0.905,0.936,0.940,1.030])) 

# VIIRS = list(np.array([0.411,0.444,0.489,0.556,0.667,0.746,0.867,1.238,1.375,1.604,
# 2.2587,
# Aqua = list(np.array([0.645,0.856,0.466,0.554,1.241,1.628,2.113,0.412,0.442,0.487,
#                 0.530,0.547,0.666,0.678,0.747,0.867,0.904,0.936,0.935]))

# i for 3D and j for VIIRS
# MatchBand = [(1,3),(2,4),(3,5),(4,7),(5,9),(6,10),(7,11),(8,1),(9,2),(10,3),
#              (11,4),(12,5),(13,8),(14,6),(15,7),(16,7),(17,7),(18,7),(19,7)]
## 可用
# MatchBandMERSI_VIIRS = [(1,3),(2,4),(3,5),(4,7),(5,9),(6,10),(8,1),(9,2),(10,3),
#              (11,4),(12,5),(13,6),(14,6),(15,7),(16,7)]

# MatchBandMERSI_MODIS = [(1,3),(2,4),(3,1),(4,2),(5,5),(6,6),(7,7),(8,8),(9,9),
#              (10,10),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),
#              (17,18),(18,19),(19,19)]
## 全部用来画光谱函数，找最近的通道
MatchBandMERSI_VIIRS = np.array([(1,3),(2,4),(3,5),(4,7),(5,8),(6,10),(7,11),(8,1),(9,2),(10,3),
             (11,4),(12,5),(13,5),(14,6),(15,7),(16,7),(17,7),(18,7),(19,7)])

MatchBandMERSI_MODIS = np.array([(1,3),(2,4),(3,1),(4,2),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),
                        (11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,19)])

dict_MERSI_VIIRS = dict(zip(MatchBandMERSI_VIIRS[:,0],MatchBandMERSI_VIIRS[:,1]))
dict_MERSI_MODIS = dict(zip(MatchBandMERSI_MODIS[:,0],MatchBandMERSI_MODIS[:,1]))
# 画SRF
# 1-10 Band

for i in range(1,20,1):
    if i <= 10:  
        
        plt.subplot(2,5,i)
        plt.plot(MERSI_CH[i-1][:,0],MERSI_CH[i-1][:,1],linestyle = 'solid',color = 'tomato',label='MERSI')
        
        band_VIIRS = dict_MERSI_VIIRS[i]
        ## dict从0开始计数，所以要减1
        plt.plot(VIIRS_CH[band_VIIRS-1][:,0],VIIRS_CH[band_VIIRS-1][:,1],linestyle = 'solid',color = 'teal',label='VIIRS')
        band_MODIS = dict_MERSI_MODIS[i]
        plt.plot(MODIS_CH[band_MODIS-1][:,0],MODIS_CH[band_MODIS-1][:,1],linestyle = 'solid',color = 'purple',label='MODIS')


        # plt.axis('equal')
        plt.title(f'MERSI {i} VIIRS {band_VIIRS} MODIS {band_MODIS}')
        plt.ylim(0,1.5)
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('SRF')
        ## Scatter 里面的label可以添加图例
        plt.legend(loc='upper right')
        plt.tight_layout(h_pad=0.1,w_pad=0)
plt.show()

for i in range(1,20,1):
    if i > 10:  
        
        plt.subplot(2,5,i-10)
        plt.plot(MERSI_CH[i-1][:,0],MERSI_CH[i-1][:,1],linestyle = 'solid',color = 'tomato',label='MERSI')
        
        band_VIIRS = dict_MERSI_VIIRS[i]
        ## dict从0开始计数，所以要减1
        plt.plot(VIIRS_CH[band_VIIRS-1][:,0],VIIRS_CH[band_VIIRS-1][:,1],linestyle = 'solid',color = 'teal',label='VIIRS')
        band_MODIS = dict_MERSI_MODIS[i]
        plt.plot(MODIS_CH[band_MODIS-1][:,0],MODIS_CH[band_MODIS-1][:,1],linestyle = 'solid',color = 'purple',label='MODIS')


        # plt.axis('equal')
        plt.title(f'MERSI {i} VIIRS {band_VIIRS} MODIS {band_MODIS}')
        plt.ylim(0,1.5)
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('SRF')
        ## Scatter 里面的label可以添加图例
        plt.legend(loc='upper right')
        plt.tight_layout(h_pad=0.1,w_pad=0)
plt.show()

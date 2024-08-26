import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import glob
import re
import matplotlib.pyplot as plt


VIIRS_SRF_Path  = r'C:/Users/1/Desktop/casestudy/data/VIIRS\VIIRS_SRF'
MODIS_SRF_Path  = r'C:/Users/1/Desktop/casestudy/data/MODIS\AquaSRF'

VIIRS_SRF_Files = glob.glob(VIIRS_SRF_Path + '/*.txt')
MODIS_SRF_Files = glob.glob(MODIS_SRF_Path + '/*.txt')

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

VIIRS_CH = list(np.arange(0, len(VIIRS_SRF_Files)) )
for i in range(0, len(VIIRS_CH)):
    VIIRS_CH[i] = np.loadtxt(VIIRS_SRF_Files[i],skiprows=4)
    VIIRS_CH[i][:,0] = 1/VIIRS_CH[i][:,0]*10000000



# Aqua = list(np.array([0.645,0.856,0.466,0.554,1.241,1.628,2.113,0.412,0.442,0.487,
#                 0.530,0.547,0.666,0.678,0.747,0.867,0.904,0.936,0.935]))
# VIIRS = list(np.array([0.641,0.867, ## I通道不用
#从这开始算
# 0.411,0.444,0.489,0.556,0.667,
# 0.746,0.867,1.238,1.375,1.604,
# 2.2587,

# i for 3D and j for VIIRS
# MatchBand = [(1,3),(2,4),(3,5),(4,7),(5,9),(6,10),(7,11),(8,1),(9,2),(10,3),
#              (11,4),(12,5),(13,8),(14,6),(15,7),(16,7),(17,7),(18,7),(19,7)]
MatchBand = [(1,5),(2,7),(3,3),(4,4),(5,8),(6,10),(7,11),(8,1),(9,2),(10,3),
             (11,4),(12,4),(13,5),(14,5),(15,6),(16,7),(17,7),(18,7),(19,7)]
# 匹配较好的通道
# MatchBand = [(1,5),(2,7),(4,4),(5,8),(6,10),(8,1),(9,2),(10,3),
#              ,(12,4),(13,5),(14,5),(15,6),(16,7),(17,7)]

# 画SRF
# 1-10 Band
k = 1
for i,j in MatchBand:
    # 1通道是列表第0个，故减去1
    i = i-1
    j = j-1
    if i < 10:
        
        plt.subplot(2,5,k)
        plt.plot(MODIS_CH[i][:,0],MODIS_CH[i][:,1],linestyle = 'solid',color = 'teal')
        plt.plot(VIIRS_CH[j][:,0],VIIRS_CH[j][:,1],linestyle = 'solid',color = 'tomato')
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('SRF')
        # plt.legend(['MERISI-II','VIIRS'])
        plt.title('MODIS %d' % (i+1)+ ' & '+'VIIRS%d' % (j+1))
        k+=1
        
plt.tight_layout(h_pad=0.1,w_pad=0)
plt.show()

# 10-19 Band
k=1
for i,j in MatchBand:
    # 1通道是列表第0个，故减去1
    i = i-1
    j = j-1
    if i >= 10:
        plt.subplot(2,5,k)
        plt.plot(MODIS_CH[i][:,0],MODIS_CH[i][:,1],linestyle = 'solid',color = 'teal')
        plt.plot(VIIRS_CH[j][:,0],VIIRS_CH[j][:,1],linestyle = 'solid',color = 'tomato')
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('SRF')
        # plt.legend(['MERISI-II','VIIRS'])
        plt.title('MODIS %d' % (i+1)+' & '+'VIIRS%d' % (j+1))
        k+=1
plt.tight_layout(h_pad=0.1,w_pad=0)
plt.show()

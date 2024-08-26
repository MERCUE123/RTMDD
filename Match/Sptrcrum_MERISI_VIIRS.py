import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import glob
import re
import matplotlib.pyplot as plt

MERSI_SRF_Path   = r'C:/Users/1/Desktop/casestudy/data/MERSI\MERSI_SRF'
VIIRS_SRF_Path  = r'C:/Users/1/Desktop/casestudy/data/VIIRS\VIIRS_SRF'

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

# MERSI = list(np.array([0.471,0.554,0.653,0.868,1.381,1.645,2.125,0.411,0.444,0.490,
#                 0.556,0.670,0.709,0.746,0.865,0.905,0.936,0.940,1.030])) 
# VIIRS = list(np.array([0.641,0.867, ## I通道不用
# 0.411,0.444,0.489,0.556,0.667,
# 0.746,0.867,1.238,1.375,1.604,
# 2.2587,
#                M  3.739,3.696,4.068,8.580]))

# i for 3D and j for VIIRS
# MatchBand = [(1,3),(2,4),(3,5),(4,7),(5,9),(6,10),(7,11),(8,1),(9,2),(10,3),
#              (11,4),(12,5),(13,8),(14,6),(15,7),(16,7),(17,7),(18,7),(19,7)]
MatchBand = [(1,3),(2,4),(3,5),(4,7),(5,9),(6,10),(8,1),(9,2),(10,3),
             (11,4),(12,5),(13,6),(14,6),(15,7),(16,7)]
MatchBand = [(1,3),(2,4),(3,1),(4,2),(5,5),(6,6),(7,7),(8,8),(9,9),
             (10,10),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),
             (17,18),(18,19),(19,19)]

# 画SRF
# 1-10 Band
k = 0
for i,j in MatchBand:
    # 1通道是列表第0个，故减去1
    i = i-1
    j = j-1
    if i < 9:
        k+=1
        plt.subplot(2,4,k)
        plt.plot(MERSI_CH[i][:,0],MERSI_CH[i][:,1],linestyle = 'solid',color = 'teal')
        plt.plot(VIIRS_CH[j][:,0],VIIRS_CH[j][:,1],linestyle = 'solid',color = 'tomato')
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('SRF')
        # plt.legend(['MERISI-II','VIIRS'])
        plt.title('MERSI-II%d' % (i+1)+ ' & '+'VIIRS%d' % (j+1))
        
plt.tight_layout(h_pad=0.1,w_pad=0)
plt.show()

# 10-19 Band
k=1
for i,j in MatchBand:
    # 1通道是列表第0个，故减去1
    i = i-1
    j = j-1
    if i >= 9:
        plt.subplot(2,4,k)
        plt.plot(MERSI_CH[i][:,0],MERSI_CH[i][:,1],linestyle = 'solid',color = 'teal')
        plt.plot(VIIRS_CH[j][:,0],VIIRS_CH[j][:,1],linestyle = 'solid',color = 'tomato')
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('SRF')
        # plt.legend(['MERISI-II','VIIRS'])
        plt.title('MERSI-II%d' % (i+1)+' & '+'VIIRS%d' % (j+1))
        k+=1
plt.tight_layout(h_pad=0.1,w_pad=0)
plt.show()

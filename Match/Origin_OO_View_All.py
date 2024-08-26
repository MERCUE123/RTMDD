import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import*
from Py6S import*
from Py6S.Params import PredefinedWavelengths, Wavelength
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
from dateutil.parser import parse
import glob
from datetime import datetime, timedelta
import pandas as pd
import os
from scipy import interpolate
import h5py as h5py
Term = 'MERSI_MODIS'
Month = 'March'
NDVIi = 0.0
Angle = 20
dTime = 300
MatchTablePathMERSI_VIIRS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MERSI_VIIRS_'+Month+'.csv'
MatchTablePathMODIS_VIIRS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MODIS_VIIRS_'+Month+'.csv'
MatchTablePathMERSI_MODIS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MERSI_MODIS_'+Month+'.csv'
match Term:
    case 'MERSI_VIIRS':
        MatchTablePath = MatchTablePathMERSI_VIIRS
    case 'MODIS_VIIRS':
        MatchTablePath = MatchTablePathMODIS_VIIRS
    case 'MERSI_MODIS':
        MatchTablePath = MatchTablePathMERSI_MODIS


sOutFile = MatchTablePath.replace('.csv','ForOrigin.csv')

## MERSI&VIIRS
MatchBandMERSI_VIIRS = np.array([(1,3),(2,4),(3,5),(4,7),(5,9),(6,10),(8,1),(9,2),(10,3),
                            (11,4),(12,5),(14,6),(15,7),(16,7)])
MatchBandMERSI_MODIS = np.array([[1,3],[2,4],[3,1],[4,2],[6,6],[7,7],[8,8],[9,9],
                            [10,10],[11,12],[12,13],[14,15],[15,16],[16,17],
                            [17,18],[18,19]])
MatchBandMODIS_VIIRS = np.array([(1,5),(2,7),(4,4),(5,8),(6,10),(8,1),(9,2),(10,3),
                            (12,4),(13,5),(14,5),(15,6),(16,7),(17,7)])

match Term:
    case 'MERSI_VIIRS':
                MatchBand = MatchBandMERSI_VIIRS

    case 'MODIS_VIIRS':
## MODIS&VIIRS
                MatchBand = MatchBandMODIS_VIIRS
    case 'MERSI_MODIS':
## MODIS&MERSI
                MatchBand = MatchBandMERSI_MODIS
#创建空csv文件
df = open(sOutFile, 'w')
df.close()

# 读取匹配数据
dp = pd.read_csv(MatchTablePath,index_col=False)

# 筛选条件
#.index获取需要删除的行索引，然后使用.drop()删除这些行
# for i in range(0,MatchBand.shape[0]):
dp = dp.drop(dp[ (dp['NDVI']<NDVIi)| (dp['dTime']<dTime) ].index,axis=0)
# dp = dp.drop(dp[(dp['VZA_MERSI']>40) & (dp['VZA_MERSI']> 40)].index,axis=0)
match Term:
        case 'MERSI_VIIRS':
            Angle_term = 'VZA_VIIRS'
        case 'MODIS_VIIRS':
            Angle_term = 'VZA_VIIRS'
        case 'MERSI_MODIS':
            Angle_term = 'VZA_MODIS'

dp = dp.drop(dp[(dp[Angle_term]>Angle) & (dp[Angle_term]> Angle)].index,axis=0)
# 匹配通道列名
# 空csv无法添加列名，加个temp列名，后面再删除  
df = pd.read_csv(sOutFile,index_col = False,names=['temp'])

match Term:
    case 'MERSI_VIIRS':
        for i in range(0,MatchBand.shape[0]):
                df.insert(df.shape[1], f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)' ,dp[f'B{MatchBand[i,0]}_MERSI'].drop(dp[(dp[f'B{MatchBand[i,1]}_VIIRS']<0.1)].index,axis=0))
                df.insert(df.shape[1], f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)_MERSI 通道{MatchBand[i,0]}' ,dp[f'B{MatchBand[i,1]}_VIIRS'].drop(dp[(dp[f'B{MatchBand[i,1]}_VIIRS']<0.1)].index,axis=0))
                df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)'] = df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)'].drop(df[(df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)_MERSI 通道{MatchBand[i,0]}']<0.1)].index,axis=0)
                df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)_MERSI 通道{MatchBand[i,0]}'] = df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)_MERSI 通道{MatchBand[i,0]}'].drop(df[(df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)_MERSI 通道{MatchBand[i,0]}']<0.1)].index,axis=0)
    
    
    case 'MODIS_VIIRS':
        for i in range(0,MatchBand.shape[0]):
    
                df.insert(df.shape[1], f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)' ,dp[f'B{MatchBand[i,0]}_MODIS'].drop(dp[(dp[f'B{MatchBand[i,1]}_MODIS']<0.1)].index,axis=0))
                df.insert(df.shape[1], f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)_MODIS 通道{MatchBand[i,0]}' ,dp[f'B{MatchBand[i,1]}_VIIRS'].drop(dp[(dp[f'B{MatchBand[i,1]}_MODIS']<0.1)].index,axis=0))
                df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)'] = df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)'].drop(df[(df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)']<0.1)].index,axis=0)
                df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)_MODIS 通道{MatchBand[i,0]}'] = df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)_MODIS 通道{MatchBand[i,0]}'].drop(df[(df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)']<0.1)].index,axis=0)
    case 'MERSI_MODIS':
            for i in range(0,MatchBand.shape[0]):
                df.insert(df.shape[1], f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)' ,dp[f'B{MatchBand[i,0]}_MERSI'].drop(dp[(dp[f'B{MatchBand[i,1]}_MODIS']<0.1)].index,axis=0))
                df.insert(df.shape[1], f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)' ,dp[f'B{MatchBand[i,1]}_MODIS'].drop(dp[(dp[f'B{MatchBand[i,1]}_MODIS']<0.1)].index,axis=0))
                df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)'] = df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m2/sr/mic)'].drop(df[(df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)']<0.1)].index,axis=0)
                df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)'] = df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)'].drop(df[(df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m2/sr/mic)']<0.1)].index,axis=0)


df.drop('temp',axis=1,inplace=True)
df.to_csv(sOutFile,index=False)

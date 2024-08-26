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
## MERSI_MODIS限制MERSI角度为35度
## 
Term = 'MODIS_VIIRS'
Month = 'June'
# Month = 'March'
NDVIi = 0.1
Angle = 60
dTime = 900
MatchTablePathMERSI_VIIRS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MERSI_VIIRS_'+Month+'.csv'
MatchTablePathMODIS_VIIRS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MODIS_VIIRS_'+Month+'.csv'
MatchTablePathMERSI_MODIS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MERSI_MODIS_'+Month+'.csv'
# MatchTablePathMERSI_MODIS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MERSI_MODIS_03_.csv'
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
                            (11,4),(12,5),(14,6),(15,7),(16,7),(17,7),(13,5)])
MatchBandMERSI_MODIS = np.array([[1,3],[2,4],[3,1],[4,2],[6,6],[7,7],[8,8],[9,9],
                            [10,10],[11,12],[12,13],[14,15],[15,16],[16,17],
                            [17,18],[18,19]])
MatchBandMODIS_VIIRS = np.array([(1,5),(2,7),(3,3),(4,4),(5,8),(6,10),(8,1),(9,2),(10,3),
                            (12,4),(13,5),(14,5),(15,6),(16,7),(17,7),(18,7)]) 

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

match Term:
        case 'MERSI_VIIRS':
            Angle_term1 = 'VZA_VIIRS'
            Angle_term2 = 'VZA_MERSI'
        case 'MODIS_VIIRS':
            Angle_term1 = 'VZA_VIIRS'
            Angle_term2 = 'VZA_MODIS'
        case 'MERSI_MODIS':
            Angle_term1 = 'VZA_MODIS'
            Angle_term2 = 'VZA_MERSI'
dp = dp.drop(dp[(dp[Angle_term1]>Angle)|(dp[Angle_term2]>Angle)].index,axis=0)
dp = dp.drop(dp[(dp['NDVI']<NDVIi)| (dp['dTime']>dTime)].index,axis=0)
print('The num of points is:',dp.shape[0])
# dp = dp.drop(dp[(dp['VAA_MERSI']<100)].index,axis=0)
# dp = dp[dp['ERA5Day'].isin([20])]
## 经纬度
# D = (dp['B8_MERSI']-dp['B1_VIIRS'])/dp['B8_MERSI']
# VAA = dp['VAA_MERSI']
# VZA = dp['VZA_MERSI']
# SZA = dp['SZA_MERSI']
# A = SZA-VZA
# NDVI=dp['NDVI']
# plt.scatter(SZA,D)
# plt.ylim(-1,1)
# plt.show()

# 匹配通道列名
# 空csv无法添加列名，加个temp列名，后面再删除  
df = pd.read_csv(sOutFile,index_col = False,names=['temp'])
match Term:
    case 'MERSI_VIIRS':
        for i in range(0,MatchBand.shape[0]):
                df.insert(df.shape[1], f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)' ,dp[f'B{MatchBand[i,0]}_MERSI'])
                df.insert(df.shape[1], f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}' ,dp[f'B{MatchBand[i,1]}_VIIRS'])
                df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'] = df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'].drop(df[(df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}']<0.1)].index,axis=0)
                df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'] = df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'].drop(df[(df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']<0.1)].index,axis=0)
                df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}'] = df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}'].drop(df[(df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']<0.1)].index,axis=0)
                df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}'] = df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}'].drop(df[(df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}']<0.1)].index,axis=0)
               

    
    
    case 'MODIS_VIIRS':
        for i in range(0,MatchBand.shape[0]):
    
                df.insert(df.shape[1], f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)' ,dp[f'B{MatchBand[i,0]}_MODIS'])
                df.insert(df.shape[1], f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}' ,dp[f'B{MatchBand[i,1]}_VIIRS'])
                df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'] = df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'].drop(df[(df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']<0.1)].index,axis=0)
                df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'] = df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'].drop(df[(df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}']<0.1)].index,axis=0)
                df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}'] = df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}'].drop(df[(df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']<0.1)].index,axis=0)
                df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}'] = df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}'].drop(df[(df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}']<0.1)].index,axis=0)
    case 'MERSI_MODIS':
            for i in range(0,MatchBand.shape[0]):
                df.insert(df.shape[1], f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)' ,dp[f'B{MatchBand[i,0]}_MERSI'])
                df.insert(df.shape[1], f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)' ,dp[f'B{MatchBand[i,1]}_MODIS'])
                df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'] = df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)'].drop(df[(df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)']<0.1)].index,axis=0)
                df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)'] = df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)'].drop(df[(df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)']<0.1)].index,axis=0)
df.drop('temp',axis=1,inplace=True)
df.to_csv(sOutFile,index=False)

match Term:
    
    case 'MERSI_MODIS':
        k=0
        for i in range(0,MatchBand.shape[0]):
            if i <= 7:
                    k+=1
                    xy0 = np.vstack([[df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']],[df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)']]])
                    xy0 = xy0.transpose()
                    ## .any(axis=1)表示某一行里面有nan就给取False
                    xy = xy0[~np.isnan(xy0).any(axis=1)]
                    # 删除nan值
                    xsort_indices = np.argsort(xy[:,0])    
                    # 按x从小到大排序
                    x = xy[xsort_indices,0]
                    y = xy[xsort_indices,1]
                    z1 = np.polyfit(x,y,2)
                    #使用次数合成多项式
                    p1 = np.poly1d(z1) 
                    y_pre = p1(x)
                    r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                    plt.subplot(2,4,k)
                    plt.plot(x,y,'+',color='blue',label='MERSI Band '+str(i))
                    plt.plot(x,y_pre,color='red',label='Fitting Curve')
                    plt.legend([f'N:{len(x)}',f'r^2={r2:.3f}'])
                    plt.xlabel(f'MERSI Band '+str(MatchBand[i,0]))
                    plt.ylabel(f'MODIS Band '+str(MatchBand[i,1]))
                    plt.xlim(0,1.2*max(x))
                    plt.ylim(0,1.2*max(y))
                    plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()
        k=0
        for i in range(0,MatchBand.shape[0]):
                if i >7:
                    k+=1
                    xy0 = np.vstack([[df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']],[df[f'MODIS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)']]])
                    xy0 = xy0.transpose()
                    ## .any(axis=1)表示某一行里面有nan就给取False
                    xy = xy0[~np.isnan(xy0).any(axis=1)]
                    # 删除nan值
                    xsort_indices = np.argsort(xy[:,0])    
                    # 按x从小到大排序
                    x = xy[xsort_indices,0]
                    y = xy[xsort_indices,1]
                    try :
                        z1 = np.polyfit(x,y,2)
                        p1 = np.poly1d(z1) #使用次数合成多项式
                        y_pre = p1(x)
                        r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                        plt.subplot(2,4,k)
                        plt.plot(x,y,'+',color='blue',label='MERSI Band '+str(i))
                        plt.plot(x,y_pre,color='red',label='Fitting Curve')
                        plt.legend([f'N:{len(x)}',f'r^2={r2:.3f}'])
                        plt.xlabel(f'MERSI Band '+str(MatchBand[i,0]))
                        plt.ylabel(f'MODIS Band '+str(MatchBand[i,1]))
                        plt.xlim(0,1.2*max(x))
                        plt.ylim(0,1.2*max(y))
                        plt.tight_layout(h_pad=0.1,w_pad=0)
                        plt.subplots_adjust(hspace=0.4, wspace=0.4)
                    except TypeError: 
                        
                        plt.scatter(0,0)
                        plt.xlabel(f'MERSI Band '+str(MatchBand[i,0]))
                        plt.ylabel(f'MODIS Band '+str(MatchBand[i,1]))

        plt.show()
        
    case 'MERSI_VIIRS':
        k=0
        for i in range(0,MatchBand.shape[0]):
            if i <= 7:
                    k+=1
                    xy0 = np.vstack([[df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']],[df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}']]])
                    xy0 = xy0.transpose()
                    ## .any(axis=1)表示某一行里面有nan就给取False
                    xy = xy0[~np.isnan(xy0).any(axis=1)]
                    # 删除nan值
                    xsort_indices = np.argsort(xy[:,0])    
                    # 按x从小到大排序
                    x = xy[xsort_indices,0]
                    y = xy[xsort_indices,1]
                    z1 = np.polyfit(x,y,2)
                    #使用次数合成多项式
                    p1 = np.poly1d(z1) 
                    y_pre = p1(x)
                    r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                    plt.subplot(2,4,k)
                    plt.plot(x,y,'+',color='blue',label='MERSI Band '+str(i))
                    plt.plot(x,y_pre,color='red',label='Fitting Curve')
                    plt.legend([f'N:{len(x)}',f'r^2={r2:.3f}'])
                    plt.xlabel(f'MERSI Band '+str(MatchBand[i,0]))
                    plt.ylabel(f'VIIRS Band '+str(MatchBand[i,1]))
                    plt.xlim(0,1.2*max(x))
                    plt.ylim(0,1.2*max(y))
                    plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()
        k=0
        for i in range(0,MatchBand.shape[0]):
                if i >7:
                    k+=1
                    xy0 = np.vstack([[df[f'MERSI 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']],[df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MERSI 通道{MatchBand[i,0]}']]])
                    xy0 = xy0.transpose()
                    ## .any(axis=1)表示某一行里面有nan就给取False
                    xy = xy0[~np.isnan(xy0).any(axis=1)]
                    # 删除nan值
                    xsort_indices = np.argsort(xy[:,0])    
                    # 按x从小到大排序
                    x = xy[xsort_indices,0]
                    y = xy[xsort_indices,1]
                    try :
                        z1 = np.polyfit(x,y,2)
                        p1 = np.poly1d(z1) #使用次数合成多项式
                        y_pre = p1(x)
                        r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                        plt.subplot(2,4,k)
                        plt.plot(x,y,'+',color='blue',label='MERSI Band '+str(i))
                        plt.plot(x,y_pre,color='red',label='Fitting Curve')
                        plt.legend([f'N:{len(x)}',f'r^2={r2:.3f}'])
                        plt.xlabel(f'MERSI Band '+str(MatchBand[i,0]))
                        plt.ylabel(f'VIIRS Band '+str(MatchBand[i,1]))
                        plt.xlim(0,1.2*max(x))
                        plt.ylim(0,1.2*max(y))
                        plt.subplots_adjust(hspace=0.4, wspace=0.4)
                    except TypeError: 
                        
                        plt.scatter(0,0)
                        plt.xlabel(f'MERSI Band '+str(MatchBand[i,0]))
                        plt.ylabel(f'VIIRS Band '+str(MatchBand[i,1]))

        plt.show()

    case 'MODIS_VIIRS':
        k=0
        for i in range(0,MatchBand.shape[0]):
            if i <= 7:
                    k+=1
                    xy0 = np.vstack([[df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']],[df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}']]])
                    xy0 = xy0.transpose()
                    ## .any(axis=1)表示某一行里面有nan就给取False
                    xy = xy0[~np.isnan(xy0).any(axis=1)]
                    # 删除nan值
                    xsort_indices = np.argsort(xy[:,0])    
                    # 按x从小到大排序
                    x = xy[xsort_indices,0]
                    y = xy[xsort_indices,1]
                    z1 = np.polyfit(x,y,2)
                    #使用次数合成多项式
                    p1 = np.poly1d(z1) 
                    y_pre = p1(x)
                    r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                    plt.subplot(2,4,k)
                    plt.plot(x,y,'+',color='blue',label='MODIS Band '+str(i))
                    plt.plot(x,y_pre,color='red',label='Fitting Curve')
                    plt.legend([f'N:{len(x)}',f'r^2={r2:.3f}'])
                    plt.xlabel(f'MODIS Band '+str(MatchBand[i,0]))
                    plt.ylabel(f'VIIRS Band '+str(MatchBand[i,1]))
                    plt.xlim(0,1.2*max(x))
                    plt.ylim(0,1.2*max(y))
                    plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()
        k=0
        for i in range(0,MatchBand.shape[0]):
                if i >7:
                    k+=1
                    xy0 = np.vstack([[df[f'MODIS 通道{MatchBand[i,0]}实际观测值(W/m^2/sr/μm)']],[df[f'VIIRS 通道{MatchBand[i,1]}实际观测值(W/m^2/sr/μm)_MODIS 通道{MatchBand[i,0]}']]])
                    xy0 = xy0.transpose()
                    ## .any(axis=1)表示某一行里面有nan就给取False
                    xy = xy0[~np.isnan(xy0).any(axis=1)]
                    # 删除nan值
                    xsort_indices = np.argsort(xy[:,0])    
                    # 按x从小到大排序
                    x = xy[xsort_indices,0]
                    y = xy[xsort_indices,1]
                    try :
                        z1 = np.polyfit(x,y,2)
                        p1 = np.poly1d(z1) #使用次数合成多项式
                        y_pre = p1(x)
                        r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                        plt.subplot(2,4,k)
                        plt.plot(x,y,'+',color='blue',label='MODIS Band '+str(MatchBand[i,0]))
                        plt.plot(x,y_pre,color='red',label='Fitting Curve')
                        plt.legend([f'N:{len(x)}',f'r^2={r2:.3f}'])
                        plt.xlabel(f'MODIS Band '+str(MatchBand[i,0]))
                        plt.ylabel(f'VIIRS Band '+str(MatchBand[i,1]))
                        plt.xlim(0,1.2*max(x))
                        plt.ylim(0,1.2*max(y))
                        plt.subplots_adjust(hspace=0.4, wspace=0.4)
                    except TypeError: 
                        
                        plt.scatter(0,0)
                        plt.xlabel(f'MODIS Band '+str(MatchBand[i,0]))
                        plt.ylabel(f'VIIRS Band '+str(MatchBand[i,1]))

        plt.show()


    

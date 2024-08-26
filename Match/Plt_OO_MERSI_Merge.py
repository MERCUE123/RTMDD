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
from Py6S.Params import PredefinedWavelengths, Wavelength
from scipy import interpolate
import h5py as h5py
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

NDVIi = 0.3
AngleVIIRS = 60
AngleMODIS = 60
Month = 'March'

MatchTablePathMERSI_VIIRS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MERSI_VIIRS_'+Month+'.csv'
MatchTablePathVIIRS_MODIS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MODIS_VIIRS_'+Month+'.csv'
MatchTablePathMERSI_MODIS = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/LandMatchTable_MERSI_MODIS_'+Month+'.csv'
## MERSI&VIIRS
MatchBandMERSI_VIIRS = np.array([(1,3),(2,4),(3,5),(4,7),(5,9),(6,10),(8,1),(9,2),(10,3),
                            (11,4),(12,5),(13,5),(14,6),(15,7),(16,7)])
MatchBandMERSI_MODIS = np.array([[1,3],[2,4],[3,1],[4,2],[6,6],[7,7],[8,8],[9,9],
                            [10,10],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],
                            [17,18],[18,19]])
MatchBandVIIRS_MODIS = np.array([(1,5),(2,7),(4,4),(5,8),(6,10),(8,1),(9,2),(10,3),
                            (12,4),(13,5),(14,5),(15,6),(16,7),(17,7)])
dict_MERSI_VIIRS = dict(zip(MatchBandMERSI_VIIRS[:,0],MatchBandMERSI_VIIRS[:,1]))
dict_MERSI_MODIS = dict(zip(MatchBandMERSI_MODIS[:,0],MatchBandMERSI_MODIS[:,1]))

def MERSI_Merge():
        Dvm = pd.read_csv(MatchTablePathMERSI_VIIRS,index_col=False)
        Dmm = pd.read_csv(MatchTablePathMERSI_MODIS,index_col=False)
        Dvm = Dvm.drop(Dvm[(Dvm.NDVI< NDVIi) | (Dvm.VZA_VIIRS> AngleVIIRS) | (Dvm.VZA_VIIRS> AngleVIIRS)].index)
        Dmm = Dmm.drop(Dmm[(Dmm.NDVI< NDVIi) | (Dmm.VZA_MERSI> AngleMODIS) | (Dmm.VZA_MODIS> AngleMODIS)].index)
        for i in range(0,MatchBandMERSI_VIIRS.shape[0]):
                        ## Drop 加个等于号
                        Dvm[f'B{MatchBandMERSI_VIIRS[i,1]}_VIIRS'] = Dvm[f'B{MatchBandMERSI_VIIRS[i,1]}_VIIRS'].drop(Dvm[(Dvm[f'B{MatchBandMERSI_VIIRS[i,1]}_VIIRS']<0.5)].index,axis=0)
                        Dvm[f'B{MatchBandMERSI_VIIRS[i,0]}_MERSI'] = Dvm[f'B{MatchBandMERSI_VIIRS[i,0]}_MERSI'].drop(Dvm[(Dvm[f'B{MatchBandMERSI_VIIRS[i,0]}_MERSI']<0.5)].index,axis=0)
        for i in range(0,MatchBandMERSI_MODIS.shape[0]):
                        Dmm[f'B{MatchBandMERSI_MODIS[i,1]}_MODIS'] = Dmm[f'B{MatchBandMERSI_MODIS[i,1]}_MODIS'].drop(Dmm[(Dmm[f'B{MatchBandMERSI_MODIS[i,1]}_MODIS']<0.5)].index,axis=0)
                        Dmm[f'B{MatchBandMERSI_MODIS[i,0]}_MERSI'] = Dmm[f'B{MatchBandMERSI_MODIS[i,0]}_MERSI'].drop(Dmm[(Dmm[f'B{MatchBandMERSI_MODIS[i,0]}_MERSI']<0.5)].index,axis=0)
        k = 1
        for i in range(1,19,1):
                if i <= 10:  
# 使用 enumerate 函数查找索引
                        for q in dict_MERSI_VIIRS.keys():
                                        if i ==q :
                                                plt.subplot(2,5,k)
                                                plt.scatter(Dvm[f'B{i}_MERSI'],Dvm[f'B{dict_MERSI_VIIRS[i]}_VIIRS'],linestyle = 'solid',color = 'r',marker='+',label='VIIRS')
                                                bandv = dict_MERSI_VIIRS[i]
                                                break
                                        else:
                                        
                                                bandv = 0

                        for w in dict_MERSI_MODIS.keys():
                                        if i ==w:
                                                plt.subplot(2,5,k)
                                                plt.scatter(Dmm[f'B{i}_MERSI'],Dmm[f'B{dict_MERSI_MODIS[i]}_MODIS'],linestyle = 'solid',color = 'b',marker='+',label='MODIS')
                                                
                                                bandm = dict_MERSI_MODIS[i]
                                                break
                                        else:
                                                bandm = 0
                                                #       plt.xlabel(f'MODIS 通道{dict_MERSI_MODIS[k][1]}实际观测值(W/m2/sr/mic)')
                        plt.axis('equal')
                        plt.xlabel(f'MERSI 通道{i}实际观测值(W/m2/sr/mic)')
                        plt.ylabel(f'VIIRS 通道{bandv} 与 MODIS通道{bandm}')
                        ## Scatter 里面的label可以添加图例
                        plt.legend()
                        k += 1

        plt.show()
        k =1
        for i in range(1,19,1):
                if i > 10:  
# 使用 enumerate 函数查找索引
                        for q in dict_MERSI_VIIRS.keys():
                                        if i ==q :
                                                plt.subplot(2,5,k)
                                                plt.scatter(Dvm[f'B{i}_MERSI'],Dvm[f'B{dict_MERSI_VIIRS[i]}_VIIRS'],linestyle = 'solid',color = 'r',marker='+',label='VIIRS')
                                                bandv = dict_MERSI_VIIRS[i]
                                                break
                                        else:
                                        
                                                bandv = 0

                        for w in dict_MERSI_MODIS.keys():
                                        if i ==w:
                                                plt.subplot(2,5,k)
                                                plt.scatter(Dmm[f'B{i}_MERSI'],Dmm[f'B{dict_MERSI_MODIS[i]}_MODIS'],linestyle = 'solid',color = 'b',marker='+',label='MODIS')
                                                
                                                bandm = dict_MERSI_MODIS[i]
                                                break
                                        else:
                                                bandm = 0
                                                #       plt.xlabel(f'MODIS 通道{dict_MERSI_MODIS[k][1]}实际观测值(W/m2/sr/mic)')
                        plt.axis('equal')
                        plt.xlabel(f'MERSI 通道{i}实际观测值(W/m2/sr/mic)')
                        plt.ylabel(f'VIIRS 通道{bandv} 与 MODIS通道{bandm}')
                        ## Scatter 里面的label可以添加图例
                        plt.legend()
                        k += 1

        plt.show()

if __name__ == '__main__':
    MERSI_Merge()
              









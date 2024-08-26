# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import sys
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl
# from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
import datetime
import pandas as pd
from scipy import interpolate
import os
from scipy.stats import gaussian_kde
plt.rcParams['font.sans-serif']=['Simsun']
plt.rcParams['axes.unicode_minus'] = False

size = 0.5
transparence = 0.5
class World_Ditribute:
    # def __init__(self):
        # self.MERSI_MODIS_matchfile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/LandMatchTable_MERSI_MODIS_'+Month+'.csv'
        # self.MERSI_VIIRS_matchfile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/LandMatchTable_MERSI_VIIRS_'+Month+'.csv'
        # self.MODIS_VIIRS_matchfile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/LandMatchTable_MODIS_VIIRS_'+Month+'.csv'
        
    def World_outline(self):
        World_outline_file = r"C:\Users\1\Desktop\casestudy\data\landfile\World.dat"
        World_outline = np.loadtxt(World_outline_file, dtype=np.float64)
        plt.subplots(figsize=(2,1),dpi=120)
        plt.scatter(World_outline[:,0], World_outline[:,1],color='black',s=0.1)
        plt.xlim(0,360)
        plt.ylim(-90,90)
        plt.xticks(np.arange(0,361,30))
        plt.yticks(np.arange(-90,91,30))
        plt.grid()
        plt.xlabel('经度/°', fontsize=14)
        plt.ylabel('纬度/°', fontsize=14)
        # plt.title("Distribute of Matchpoint ", fontsize=16)
           
        # plt.show()
    def MERSI_MODIS_Distribute(self,Month):
        MERSI_MODIS_matchfile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/LandMatchTable_MERSI_MODIS_'+Month+'.csv'
        
        Matchtable = pd.read_csv(MERSI_MODIS_matchfile,index_col=False)
        Matchtable = Matchtable.drop(Matchtable[Matchtable.NDVI < 0.1].index)
        lontitude = 0.25 * Matchtable['jPos']
        latitude = 90 - 0.25 * Matchtable['iPos']
        return lontitude,latitude
    def MERSI_VIIRS_Distribute(self,Month):
        MERSI_VIIRS_matchfile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/LandMatchTable_MERSI_VIIRS_'+Month+'.csv'
        set_color = plt.cm.tab10(1)
        Matchtable = pd.read_csv(MERSI_VIIRS_matchfile,index_col=False)
        Matchtable = Matchtable.drop(Matchtable[Matchtable['NDVI'] <= 0.1].index)
        lontitude = 0.25 * Matchtable['jPos']
        latitude = 90 - 0.25 * Matchtable['iPos']
        return lontitude,latitude
    def MODIS_VIIRS_Distribute(self,Month):
        MODIS_VIIRS_matchfile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/LandMatchTable_MODIS_VIIRS_'+Month+'.csv'
        Matchtable = pd.read_csv(MODIS_VIIRS_matchfile,index_col=False)
        Matchtable = Matchtable.drop(Matchtable[Matchtable['NDVI'] <= 0.1].index)
        lontitude = 0.25 * Matchtable['jPos']
        latitude = 90 - 0.25 * Matchtable['iPos']
        return lontitude,latitude

    
    def Draw_MERSI_MODIS(self):
        ## MERSI_MODIS
        ## March
        lontitude,latitude = WD.MERSI_MODIS_Distribute('March')
        set_color = plt.cm.Dark2(0)
        plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'March',alpha = transparence)
        ## June
        lontitude,latitude = WD.MERSI_MODIS_Distribute('June')
        set_color = plt.cm.Dark2(1)
        plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'June',alpha = transparence)
        ## September
        lontitude,latitude = WD.MERSI_MODIS_Distribute('September')
        set_color = plt.cm.Dark2(2)
        plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'September',alpha = transparence)
        
    def Draw_MERSI_MODIS_Density(self):
        lontitude3,latitude3 = WD.MERSI_MODIS_Distribute('March')
        lontitude6,latitude6 = WD.MERSI_MODIS_Distribute('June')
        lontitude = np.hstack([lontitude3,lontitude6])
        latitude = np.hstack([latitude3,latitude6])
        xy_density = np.vstack([lontitude, latitude])
        z_density = gaussian_kde(xy_density)(xy_density)
        z_density = z_density*(len(latitude))
        idx = z_density.argsort() #对z值进行从小到大排序并返回索引
        lontitude , latitude, z_density = lontitude [idx], latitude[idx], z_density[idx]#对x,y按照z的升序进行排列
        #上面两行代码是为了使得z值越高的点，画在上面，不被那些z值低的点挡住，从美观的角度来说还是十分必要的
        norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
        sc = plt.scatter(lontitude,latitude,c=z_density,cmap='viridis',alpha=0.7,marker='.',norm=norm,s=12,linewidth =0.7)
        plt.colorbar(sc, label='Num',orientation='vertical',fraction=0.05, pad=0.05,location='right')

       
        
        
    def Draw_MERSI_VIIRS(self):
        lontitude,latitude = WD.MERSI_VIIRS_Distribute('March')
        set_color = plt.cm.Dark2(0)
        plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'March',alpha = transparence)
       
        ## June
        lontitude,latitude = WD.MERSI_VIIRS_Distribute('June')
        set_color = plt.cm.Dark2(1)
        plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'June',alpha = transparence)
        ## September
        # lontitude,latitude = WD.MERSI_VIIRS_Distribute('September')
        # set_color = plt.cm.Dark2(2)
        # plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'September',alpha = transparence)
    
    def Draw_MODIS_VIIRS(self):
        lontitude,latitude = WD.MODIS_VIIRS_Distribute('March')
        set_color = plt.cm.Dark2(0)
        plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'March',alpha = transparence)
        ## June
        lontitude,latitude = WD.MODIS_VIIRS_Distribute('June')
        set_color = plt.cm.Dark2(1)
        plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'June',alpha = transparence)
        ## September
        # lontitude,latitude = WD.MODIS_VIIRS_Distribute('September')
        # set_color = plt.cm.Dark2(2)
        # plt.scatter(lontitude, latitude, color=set_color,s=size,label = 'September',alpha = transparence)
        
    
if __name__ == '__main__':
    WD = World_Ditribute()
    WD.World_outline()

    
    ## MERSI_MODIS
    # WD.Draw_MERSI_MODIS()

    ## MERSI_VIIRS
    WD.Draw_MERSI_MODIS_Density()
    ## MODIS_VIIRS
    # WD.Draw_MODIS_VIIRS()

    
    
    
    
    # plot_All
    plt.legend(loc = 'upper right',fontsize=12) 
    plt.show()
    

        
        
        
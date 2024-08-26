import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import pandas as pd
import os
import h5py as h5py
from tqdm import tqdm
from scipy.stats import gaussian_kde
import seaborn as sns
plt.rcParams['font.sans-serif']=['Simsun']
plt.rcParams['axes.unicode_minus'] = False

color1 = plt.cm.tab10(3)
color2 = plt.cm.tab10(0)
Month = 'March'

## 是否重新匹配
retryMODIS = 1
retryVIIRS = 1
## 丢弃通道 为MODIS或VIIRS对应的MODIS通道
DropMODISBand = [5,13,14,15,16]
DropMODISBand = [lambda x:x-1 for x in DropMODISBand]
DropVIIRSBand = [16,17,18]
DropVIIRSBand = [lambda x:x-1 for x in DropVIIRSBand]

MERSIPy6sFile_With_MODIS =r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/Py6s_MERSI_'+Month+'_All_With_MODIS.csv'
MERSIPy6sFile_With_VIIRS =r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/Py6s_MERSI_'+Month+'_All_With_VIIRS.csv'
MODISPy6sFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/Py6s_MODIS_'+Month+'_All.csv'
VIIRSPy6sFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/Py6s_VIIRS_'+Month+'_All.csv'

## 创建输出表
DD_MERSI_MODISOutFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/DDFile/MERSI_MODIS_DD_MERSI_MODIS.csv'
DD_MERSI_VIIRSOutFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/DDFile/MERSI_VIIRS_DD_MERSI_VIIRS.csv'
DD_MergedOutFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/DDFile/MERSI_DD_Merged.csv'
DD_Merged = open(DD_MergedOutFile, 'w')
DD_Merged.close()
DD_Merged  = pd.read_csv(DD_MergedOutFile,index_col=False,names=['temp'])
NDVImin = 0.1
NDVImax = 0.3
SZA = 60
VZA = 60

ORD_SRD_Tempfile_Path = r'C:\Users\1\Desktop\casestudy\data\MatchRes\DDFile'

# MatchTablePath= SRD_MERSI_MODISPath
# sOutFile = MatchTablePath.replace('.csv','SS_ForOrigin.csv')

MatchBand_MERSI_MODIS = np.array([(1,3),(2,4),(3,1),(4,2),(5,5),(6,6),(7,7),(8,8),(9,9),
             (10,10),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),
             (17,18),(18,19),(19,19)])

MatchBand_MERSI_VIIRS = np.array([(1,3),(2,4),(3,5),(4,7),(5,8),(6,10),(7,11),(8,1),(9,2),(10,3),
             (11,4),(12,5),(13,5),(14,6),(15,7),(16,7),(17,7),(18,7),(19,7)])

dpMODIS = pd.read_csv(MODISPy6sFile,index_col=False)
dpVIIRS = pd.read_csv(VIIRSPy6sFile,index_col=False)
dpMERSI_MODIS = pd.read_csv(MERSIPy6sFile_With_MODIS,index_col=False)
dpMERSI_VIIRS = pd.read_csv(MERSIPy6sFile_With_VIIRS,index_col=False)


## 数据清洗，去除0值和无效值
## MERSI MODIS 
for k in range(0,MatchBand_MERSI_MODIS.shape[0]):
    ## 无效值置0
    # dpMODIS.replace(np.nan, 0, inplace=True)
    # dpMERSI_MODIS.replace(np.nan, 0, inplace=True)
    # dpMERSI_VIIRS.replace(np.nan, 0, inplace=True)
    ## 去除0值和无效值  但还得让nan占位，确保行对齐
    # print(max(dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o']),min(dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o']))
    ## 去除MODIS0值
    dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}s'] = dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}s'].drop(dpMODIS[(dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o']<0.1)].index,axis=0)
    dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o'] = dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o'].drop(dpMODIS[(dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o']<0.1)].index,axis=0)
    dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}s'] = dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}s'].drop(dpMODIS[(dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o']<0.1)].index,axis=0)
    dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}o'] = dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}o'].drop(dpMODIS[(dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o']<0.1)].index,axis=0)
    ## 去除MERSI0值
    dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}s'] = dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}s'].drop(dpMERSI_MODIS[(dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}o']<0.1)].index,axis=0)
    dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o'] = dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o'].drop(dpMERSI_MODIS[(dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}o']<0.1)].index,axis=0)
    dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}s'] = dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}s'].drop(dpMERSI_MODIS[(dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}o']<0.1)].index,axis=0)
    dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}o'] = dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}o'].drop(dpMERSI_MODIS[(dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[k,0]}o']<0.1)].index,axis=0)
    # print(max(dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o']),min(dpMODIS[f'B{MatchBand_MERSI_MODIS[k,1]}o']))


## MERSI VIIRS
for k in range(0,MatchBand_MERSI_VIIRS.shape[0]):
    ## 无效值置0
    # dpVIIRS.replace(np.nan, 0, inplace=True)
    # dpMERSI_VIIRS.replace(np.nan, 0, inplace=True)
    # dpMERSI_VIIRS.replace(np.nan, 0, inplace=True)
    ## 去除0值和无效值  但还得让nan占位，确保行对齐
    ## 去除VIIRS0值
    dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}s'] = dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}s'].drop(dpVIIRS[(dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}o']<0.1)].index,axis=0)
    dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}o'] = dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}o'].drop(dpVIIRS[(dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}o']<0.1)].index,axis=0)
    dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}s'] = dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}s'].drop(dpVIIRS[(dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}o']<0.1)].index,axis=0)
    dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}o'] = dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}o'].drop(dpVIIRS[(dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}o']<0.1)].index,axis=0)
    ## 去除MERSI0值
    dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}s'] = dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}s'].drop(dpMERSI_VIIRS[(dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}o']<0.1)].index,axis=0)
    dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}o'] = dpVIIRS[f'B{MatchBand_MERSI_VIIRS[k,1]}o'].drop(dpMERSI_VIIRS[(dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}o']<0.1)].index,axis=0)
    dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}s'] = dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}s'].drop(dpMERSI_VIIRS[(dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}o']<0.1)].index,axis=0)
    dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}o'] = dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}o'].drop(dpMERSI_VIIRS[(dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[k,0]}o']<0.1)].index,axis=0)

print(min(dpMERSI_MODIS['B6o']))

## 数据筛选，NDVI,SZA等
dpMERSI_MODIS= dpMERSI_MODIS.drop(dpMERSI_MODIS[(dpMERSI_MODIS.NDVI< NDVImin) |(dpMERSI_MODIS.NDVI> NDVImax) | (dpMERSI_MODIS.SZA> SZA) | (dpMERSI_MODIS.VZA> VZA)].index)
dpMERSI_VIIRS = dpMERSI_VIIRS.drop(dpMERSI_VIIRS[(dpMERSI_VIIRS.NDVI< NDVImin)|(dpMERSI_VIIRS.NDVI> NDVImax) | (dpMERSI_VIIRS.SZA> SZA) | (dpMERSI_VIIRS.VZA> VZA)].index)
dpMODIS = dpMODIS.drop(dpMODIS[(dpMODIS.NDVI< NDVImin)|(dpMODIS.NDVI> NDVImax) | (dpMODIS.SZA> SZA) | (dpMODIS.VZA> VZA)].index)
dpVIIRS = dpVIIRS.drop(dpVIIRS[(dpVIIRS.NDVI< NDVImin)|(dpVIIRS.NDVI> NDVImax) | (dpVIIRS.SZA> SZA) | (dpVIIRS.VZA> VZA)].index)
print(min(dpMERSI_MODIS['B6o']))
## 重置索引
dpMERSI_MODIS.reset_index(drop=True,inplace=True)
dpMERSI_VIIRS.reset_index(drop=True,inplace=True)
dpMODIS.reset_index(drop=True,inplace=True)
dpVIIRS.reset_index(drop=True,inplace=True)




def py6s_Merged_plot_part1():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i <=9:
            ## 不在MODIS中
                if i in DropMODISBand:
                        # 不在VIIRS中
                        if i in DropVIIRSBand:pass
                        # 在VIIRS中
                        else:
                                xy0 = np.vstack([dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[i,0]}o'],dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[i,0]}s']])
                                plt.subplot(2,5,i+1)    
                                xy0 = xy0.transpose()
                                # 删除nan值
                                xy = xy0[~np.isnan(xy0).any(axis=1)]
                                # 按x从小到大排序
                                xy = xy[np.argsort(xy[:,0])]    
                                x = xy[:,0]
                                y = xy[:,1]
                                # print(max(x))
                                z1 = np.polyfit(x,y,deg=2)
                                p1 =  np.poly1d(z1) #使用次数合成多项式
                                y_pre = p1(x)
                                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                ## 画密度曲线
                                xy_density = np.vstack([x, y])
                                z_density = gaussian_kde(xy_density)(xy_density)
                
                ## 在MODIS中
                else:
                        # 不在VIIRS中
                        if i  in DropVIIRSBand:
                                xy0 =np.vstack([dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[i,0]}o'],dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[i,0]}s']])
                                plt.subplot(2,5,i+1)    
                                xy0 = xy0.transpose()
                                # 删除nan值
                                xy = xy0[~np.isnan(xy0).any(axis=1)]
                                # 按x从小到大排序
                                xy = xy[np.argsort(xy[:,0])]    
                                x = xy[:,0]
                                y = xy[:,1]
                                # print(max(x))
                                z1 = np.polyfit(x,y,deg=2)
                                p1 =  np.poly1d(z1) #使用次数合成多项式
                                y_pre = p1(x)
                                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                ## 画密度曲线
                                xy_density = np.vstack([x, y])
                                z_density = gaussian_kde(xy_density)(xy_density)
                        ## 两者都有
                        else:
                                MERSI_VIIRS = np.vstack([dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[i,0]}o'],dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[i,0]}s']])
                                MERSI_MODIS = np.vstack([dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[i,0]}o'],dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[i,0]}s']])
                                xy0 = np.hstack([MERSI_VIIRS,MERSI_MODIS])
                                edge = len(MERSI_VIIRS[0])
                                plt.subplot(2,5,i+1)    
                                xy0 = xy0.transpose()
                                
                                # 删除nan值
                                xy0_VIIRS=xy0[0:edge]
                                xy0_MODIS = xy0[edge:]
                                xy_VIIRS = xy0_VIIRS[~np.isnan(xy0_VIIRS).any(axis=1)]
                                xy_MODIS = xy0_MODIS[~np.isnan(xy0_MODIS).any(axis=1)]
                                xy = xy0[~np.isnan(xy0).any(axis=1)]
                                
                                o_VIIRS = xy_VIIRS[:,0]
                                s_VIIRS = xy_VIIRS[:,1]
                                o_MODIS = xy_MODIS[:,0]
                                s_MODIS = xy_MODIS[:,1]
                                # 按x从小到大排序
                                xy_arg = xy[np.argsort(xy[:,0])]    
                                x = xy_arg[:,0]
                                y = xy_arg[:,1]
                                # print(max(x))
                                z1 = np.polyfit(x,y,deg=2)
                                p1 =  np.poly1d(z1) #使用次数合成多项式
                                y_pre = p1(x)
                                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                ## 画密度曲线
                                # xy_density_VIIRS = np.vstack([o_VIIRS, s_VIIRS])
                                # z_density_VIIRS = gaussian_kde(xy_density_VIIRS)(xy_density_VIIRS)
                                # xy_density_MODIS = np.vstack([o_MODIS, s_MODIS])
                                # z_density_MODIS = gaussian_kde(xy_density_MODIS)(xy_density_MODIS)
                
        #     range_y = max(y)-min(y)
        #     plt.ylim(-3*range_y,3*range_y)
                ## jet terrain inferno
                if i in DropMODISBand:
                # 不在VIIRS中
                        if i in DropVIIRSBand:pass
                # 在VIIRS中
                        else:   
                                norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                                plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7,label='VIIRS')
                ## 在MODIS中
                else:
                # 不在VIIRS中
                        if i in DropVIIRSBand:
                                norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                                plt.scatter(x,y,c=z_density,cmap='terrain',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7,label='MODIS')
                        ## 两者都有
                        else:   
                                # norm1 = mpl.colors.Normalize(vmin=z_density_VIIRS.min(), vmax=z_density_VIIRS.max())
                                # norm2 = mpl.colors.Normalize(vmin=z_density_MODIS.min(), vmax=z_density_MODIS.max())
                                
                                plt.scatter(o_VIIRS,s_VIIRS,color = 'red',cmap='plasma',alpha=0.4,marker='+',s=12,linewidth =0.7,label='VIIRS')
                                plt.scatter(o_MODIS,s_MODIS,color = 'blue',cmap='terrain',alpha=0.4,marker='+',s=12,linewidth =0.7,label='MODIS')
                        
                plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                plt.legend([f'VIIRS {len(o_VIIRS)}',f'MODIS {len(o_MODIS)}',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 模拟值(W/m^2/sr/μm)')
                plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')
            # plt.title(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS')
                
def py6s_Merged_plot_part2():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i >=10:
            ## 不在MODIS中
                if i in DropMODISBand:
                        # 不在VIIRS中
                        if i in DropVIIRSBand:pass
                        # 在VIIRS中
                        else:
                                xy0 = np.vstack([dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[i,0]}o'],dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[i,0]}s']])
                                plt.subplot(2,5,i-9)    
                                xy0 = xy0.transpose()
                                # 删除nan值
                                xy = xy0[~np.isnan(xy0).any(axis=1)]
                                # 按x从小到大排序
                                xy = xy[np.argsort(xy[:,0])]    
                                x = xy[:,0]
                                y = xy[:,1]
                                # print(max(x))
                                z1 = np.polyfit(x,y,deg=2)
                                p1 =  np.poly1d(z1) #使用次数合成多项式
                                y_pre = p1(x)
                                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                ## 画密度曲线
                                xy_density = np.vstack([x, y])
                                z_density = gaussian_kde(xy_density)(xy_density)
                
                ## 在MODIS中
                else:
                        # 不在VIIRS中
                        if i  in DropVIIRSBand:
                                xy0 =np.vstack([dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[i,0]}o'],dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[i,0]}s']])
                                plt.subplot(2,5,i-9)    
                                xy0 = xy0.transpose()
                                # 删除nan值
                                xy = xy0[~np.isnan(xy0).any(axis=1)]
                                # 按x从小到大排序
                                xy = xy[np.argsort(xy[:,0])]    
                                x = xy[:,0]
                                y = xy[:,1]
                                # print(max(x))
                                z1 = np.polyfit(x,y,deg=2)
                                p1 =  np.poly1d(z1) #使用次数合成多项式
                                y_pre = p1(x)
                                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                ## 画密度曲线
                                xy_density = np.vstack([x, y])
                                z_density = gaussian_kde(xy_density)(xy_density)
                        ## 两者都有
                        else:
                                MERSI_VIIRS = np.vstack([dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[i,0]}o'],dpMERSI_VIIRS[f'B{MatchBand_MERSI_VIIRS[i,0]}s']])
                                MERSI_MODIS = np.vstack([dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[i,0]}o'],dpMERSI_MODIS[f'B{MatchBand_MERSI_MODIS[i,0]}s']])
                                xy0 = np.hstack([MERSI_VIIRS,MERSI_MODIS])
                                edge = len(MERSI_VIIRS[0])
                                plt.subplot(2,5,i-9)    
                                xy0 = xy0.transpose()
                                
                                # 删除nan值
                                xy0_VIIRS=xy0[0:edge]
                                xy0_MODIS = xy0[edge:]
                                xy_VIIRS = xy0_VIIRS[~np.isnan(xy0_VIIRS).any(axis=1)]
                                xy_MODIS = xy0_MODIS[~np.isnan(xy0_MODIS).any(axis=1)]
                                xy = xy0[~np.isnan(xy0).any(axis=1)]
                                
                                o_VIIRS = xy_VIIRS[:,0]
                                s_VIIRS = xy_VIIRS[:,1]
                                o_MODIS = xy_MODIS[:,0]
                                s_MODIS = xy_MODIS[:,1]
                                # 按x从小到大排序
                                xy_arg = xy[np.argsort(xy[:,0])]    
                                x = xy_arg[:,0]
                                y = xy_arg[:,1]
                                # print(max(x))
                                z1 = np.polyfit(x,y,deg=2)
                                p1 =  np.poly1d(z1) #使用次数合成多项式
                                y_pre = p1(x)
                                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                ## 画密度曲线
                                # xy_density_VIIRS = np.vstack([o_VIIRS, s_VIIRS])
                                # z_density_VIIRS = gaussian_kde(xy_density_VIIRS)(xy_density_VIIRS)
                                # xy_density_MODIS = np.vstack([o_MODIS, s_MODIS])
                                # z_density_MODIS = gaussian_kde(xy_density_MODIS)(xy_density_MODIS)
                
        #     range_y = max(y)-min(y)
        #     plt.ylim(-3*range_y,3*range_y)
                ## jet terrain inferno
                if i in DropMODISBand:
                # 不在VIIRS中
                        if i in DropVIIRSBand:pass
                # 在VIIRS中
                        else:   
                                norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                                plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7,label='VIIRS')
                ## 在MODIS中
                else:
                # 不在VIIRS中
                        if i in DropVIIRSBand:
                                norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                                plt.scatter(x,y,c=z_density,cmap='terrain',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7,label='MODIS')
                        ## 两者都有
                        else:   
                                # norm1 = mpl.colors.Normalize(vmin=z_density_VIIRS.min(), vmax=z_density_VIIRS.max())
                                # norm2 = mpl.colors.Normalize(vmin=z_density_MODIS.min(), vmax=z_density_MODIS.max())
                                
                                plt.scatter(o_VIIRS,s_VIIRS,color = 'red',cmap='plasma',alpha=0.4,marker='+',s=12,linewidth =0.7,label='VIIRS')
                                plt.scatter(o_MODIS,s_MODIS,color = 'blue',cmap='terrain',alpha=0.4,marker='+',s=12,linewidth =0.7,label='MODIS')
                        
                plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                plt.legend([f'VIIRS {len(o_VIIRS)}',f'MODIS {len(o_MODIS)}',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 模拟值(W/m^2/sr/μm)')
                plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')
    
if __name__ == '__main__':
    # DD_DD_MERSI_MODIS_plot_part1()
    # DD_MERSI_MODIS_plot_part1()
#     py6s_Merged_plot_part1()
#     plt.show()
    py6s_Merged_plot_part2()
    plt.show()
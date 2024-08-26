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
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

color1 = plt.cm.tab10(3)
color2 = plt.cm.tab10(0)
Month = 'March'

## 是否重新匹配
retryMODIS = 0
retryVIIRS = 0
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
NDVImin = 0.05
NDVImax = 0.25
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

# MERSI MODIS SRD_MERSI_MODIS +Num
npsrd = np.full([1,MatchBand_MERSI_MODIS.shape[0]*3], np.nan)
nptemp = np.zeros((1,MatchBand_MERSI_MODIS.shape[0]*3))
nptemp2 = np.zeros((1,MatchBand_MERSI_MODIS.shape[0]*3))
npord = np.full([1,MatchBand_MERSI_MODIS.shape[0]*3], np.nan)

##################### index 被去掉了！！！！！！！！！！！！！！
## 寻找Num相同，即对应相同点的MERSI和MODIS点
if retryMODIS ==1:
    DD_MERSI_MODIS = open(DD_MERSI_MODISOutFile, 'w')
    DD_MERSI_MODIS.close()
    DD_MERSI_MODIS = pd.read_csv(DD_MERSI_MODISOutFile,index_col=False,names=['temp'])
    loc = 0
    for p,q in tqdm(enumerate(dpMERSI_MODIS['Num'])):
        
        for s,t in enumerate(dpMODIS['Num']):
            if q == t :
                for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
                        nptemp[loc,i*3 ]  = dpMERSI_MODIS.loc[p,f'B{MatchBand_MERSI_MODIS[i,0]}s']
                        nptemp[loc,i*3+1] = dpMODIS.loc[s,f'B{MatchBand_MERSI_MODIS[i,1]}s']
                        nptemp[loc,i*3+2] = dpMERSI_MODIS.loc[p,f'B{MatchBand_MERSI_MODIS[i,0]}s']-dpMODIS.loc[s,f'B{MatchBand_MERSI_MODIS[i,1]}s']
                        nptemp2[loc,i*3] = dpMERSI_MODIS.loc[p,f'B{MatchBand_MERSI_MODIS[i,0]}o']
                        nptemp2[loc,i*3+1] = dpMODIS.loc[s,f'B{MatchBand_MERSI_MODIS[i,1]}o']
                        nptemp2[loc,i*3+2] = dpMERSI_MODIS.loc[p,f'B{MatchBand_MERSI_MODIS[i,0]}o']-dpMODIS.loc[s,f'B{MatchBand_MERSI_MODIS[i,1]}o']
                npsrd = np.concatenate((npsrd,nptemp),axis=0)
                npord = np.concatenate((npord,nptemp2),axis=0)
            if t> q:
                break
            else:
                continue
            
    SRD_MERSI_MODIS = list(range(0,MatchBand_MERSI_MODIS.shape[0]))
    ORD_MERSI_MODIS = list(range(0,MatchBand_MERSI_MODIS.shape[0]))
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        SRD_MERSI_MODIS[i] = npsrd[:,i*3+2]
        ORD_MERSI_MODIS[i] = npord[:,i*3+2]
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        DD_MERSI_MODIS.insert(DD_MERSI_MODIS.shape[1],f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)',npord[:,i*3])
        DD_MERSI_MODIS.insert(DD_MERSI_MODIS.shape[1], f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS' ,ORD_MERSI_MODIS[i]-SRD_MERSI_MODIS[i])
    DD_MERSI_MODIS.drop('temp',axis=1,inplace=True) 
    DD_MERSI_MODIS.to_csv(DD_MERSI_MODISOutFile,index=False,encoding='utf-8-sig')
else:
    DD_MERSI_MODIS = pd.read_csv(DD_MERSI_MODISOutFile,index_col=False)
# 删除最开始创建的空行





# MERSI VIIRS SRD_MERSI_MODIS +Num
npsrd = np.full([1,MatchBand_MERSI_VIIRS.shape[0]*3], np.nan)
nptemp = np.zeros((1,MatchBand_MERSI_VIIRS.shape[0]*3))
nptemp2 = np.zeros((1,MatchBand_MERSI_VIIRS.shape[0]*3))
npord = np.full([1,MatchBand_MERSI_VIIRS.shape[0]*3], np.nan)

##################### index 被去掉了！！！！！！！！！！！！！！
## 寻找Num相同，即对应相同点的MERSI和VIIRS点
if retryVIIRS ==1:
    DD_MERSI_VIIRS = open(DD_MERSI_VIIRSOutFile, 'w')
    DD_MERSI_VIIRS.close()
    DD_MERSI_VIIRS = pd.read_csv(DD_MERSI_VIIRSOutFile,index_col=False,names=['temp'])
    loc = 0
    for p,q in tqdm(enumerate(dpMERSI_VIIRS['Num'])):
        
        for s,t in enumerate(dpVIIRS['Num']):
            if q == t :
                for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
                        nptemp[loc,i*3 ]  = dpMERSI_VIIRS.loc[p,f'B{MatchBand_MERSI_VIIRS[i,0]}s']
                        nptemp[loc,i*3+1] = dpVIIRS.loc[s,f'B{MatchBand_MERSI_VIIRS[i,1]}s']
                        nptemp[loc,i*3+2] = dpMERSI_VIIRS.loc[p,f'B{MatchBand_MERSI_VIIRS[i,0]}s']-dpVIIRS.loc[s,f'B{MatchBand_MERSI_VIIRS[i,1]}s']
                        nptemp2[loc,i*3] = dpMERSI_VIIRS.loc[p,f'B{MatchBand_MERSI_VIIRS[i,0]}o']
                        nptemp2[loc,i*3+1] = dpVIIRS.loc[s,f'B{MatchBand_MERSI_VIIRS[i,1]}o']
                        nptemp2[loc,i*3+2] = dpMERSI_VIIRS.loc[p,f'B{MatchBand_MERSI_VIIRS[i,0]}o']-dpVIIRS.loc[s,f'B{MatchBand_MERSI_VIIRS[i,1]}o']
                npsrd = np.concatenate((npsrd,nptemp),axis=0)
                npord = np.concatenate((npord,nptemp2),axis=0)
            if t> q:
                break
            else:
                continue
        
    npsrd = np.delete(npsrd,0,axis=0)
    npord = np.delete(npord,0,axis=0)
    SRD_MERSI_VIIRS = list(range(0,MatchBand_MERSI_VIIRS.shape[0]))
    ORD_MERSI_VIIRS = list(range(0,MatchBand_MERSI_VIIRS.shape[0]))
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        SRD_MERSI_VIIRS[i] = npsrd[:,i*3+2]
        ORD_MERSI_VIIRS[i] = npord[:,i*3+2]
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
            DD_MERSI_VIIRS.insert(DD_MERSI_VIIRS.shape[1], f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS' ,ORD_MERSI_VIIRS[i]-SRD_MERSI_VIIRS[i])
            DD_MERSI_VIIRS.insert(DD_MERSI_VIIRS.shape[1],f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)',npord[:,i*3])
    DD_MERSI_VIIRS.drop('temp',axis=1,inplace=True) 
    DD_MERSI_VIIRS.to_csv(DD_MERSI_VIIRSOutFile,index=False,encoding='utf-8-sig')
else:
    DD_MERSI_VIIRS = pd.read_csv(DD_MERSI_VIIRSOutFile,index_col=False)


####################################################################################################################################################3
####################################################################################################################################################3
####################################################################################################################################################3
#DD_MERSI_MODIS plot ####################################
def DD_MERSI_MODIS_plot_part1():
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i <=9:
            
            plt.subplot(2,5,i+1)
            if i in DropMODISBand:
                continue
            plt.plot(DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)'],DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS'],'+',color = color1)
            ## 画r^2
            xy0 = np.vstack([[DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']]])
            xy0 = xy0.transpose()
            # 删除nan值
            xy = xy0[~np.isnan(xy0).any(axis=1)]
            # 按x从小到大排序
            xy = xy[np.argsort(xy[:,0])]    
            x = xy[:,0]
            y = xy[:,1]
            z1 = np.polyfit(x,y,deg=2)
            p1 =  np.poly1d(z1) #使用次数合成多项式
            y_pre = p1(x)
            r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)

            ## 画密度曲线
            xy_density = np.vstack([x, y])
            z_density = gaussian_kde(xy_density)(xy_density)
            norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())

            range_y = max(y)-min(y)
            plt.ylim(-3*range_y,3*range_y)
            ## jet terrain inferno
            plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
            plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
            plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)')
            plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')
            # plt.title(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS')

def DD_MERSI_MODIS_plot_part2():
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i >9:
            if i in DropMODISBand:
                continue
            plt.subplot(2,5,i-9)
            
            plt.plot(DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)'],DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS'],'+',color = color1)
            ## 画r^2
            xy0 = np.vstack([[DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']]])
            xy0 = xy0.transpose()
            # 删除nan值
            xy = xy0[~np.isnan(xy0).any(axis=1)]
            # 按x从小到大排序
            xy = xy[np.argsort(xy[:,0])]    
            x = xy[:,0]
            y = xy[:,1]
            ## 拟合二项式
            try :
                z1 = np.polyfit(x,y,deg=2)
                p1 =  np.poly1d(z1) #使用次数合成多项式
                y_pre = p1(x)
                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                
                ## 画密度曲线
                xy_density = np.vstack([x, y])
                z_density = gaussian_kde(xy_density)(xy_density)
                norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                ## 画图
                range_y = max(y)-min(y)
                plt.ylim(-3*range_y,3*range_y)
                ## jet terrain inferno
                plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)')
                plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')
            except TypeError:
                plt.scatter(0,0,alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)')
                plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')

            # plt.title(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS')

#SRD_MERSI_MODIS plot ####################################
def SRD_DD_MERSI_MODIS_plot_part1():
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i <=9:
            if i in DropMODISBand:
                continue
            plt.subplot(2,5,i+1)
            plt.hist(npsrd[:,i*3+2],color = color1,align='mid',bins=10)
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} SRD_MERSI_MODIS')
            plt.ylabel('频率')
            # plt.title(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} SRD_MERSI_MODIS')
    plt.show()

    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i >9:
            if i in DropMODISBand:
                continue
            plt.subplot(2,5,i-9)
            try :
                plt.hist(npsrd[:,i*3+2],color = color1,align='mid',bins=10)
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} SRD_MERSI_MODIS')
                plt.ylabel('频率')
                # plt.title(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} SRD_MERSI_MODIS')
            except:
                ValueError

    plt.show()

#ORD_MERSI_MODIS plot ####################################
def ORD_MERSI_MODIS_plot_part1():
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i <=9:
            if i in DropMODISBand:
                continue
            plt.subplot(2,5,i+1)
            plt.hist(npord[:,i*3+2],color = color1,align='mid',bins=10)
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} ORD_MERSI_MODIS')
            plt.ylabel('频率')
            # plt.title(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} ORD_MERSI_MODIS')

def ORD_MERSI_MODIS_plot_part2():
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i >9:
            if i in DropMODISBand:
                continue
            plt.subplot(2,5,i-9)
            try:
                plt.hist(npord[:,i*3+2],color = color1,align='mid',bins=10)
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} ORD_MERSI_MODIS')
                plt.ylabel('频率')
                # plt.title(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} ORD_MERSI_MODIS')
            except:
                ValueError

####################################################################################################################################################3
####################################################################################################################################################3
#####################################################################################################################################################3
## MERSI VIIRS plot
#DD_MERSI_VIIRS plot ####################################
def DD_MERSI_VIIRS_plot_part1():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i <=9:
            if i in DropVIIRSBand:
                continue

            plt.subplot(2,5,i+1)
            plt.plot(DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)'],DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS'],'+',color = color1)
            ## 画r^2
            xy0 = np.vstack([[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']]])
            xy0 = xy0.transpose()
            # 删除nan值
            xy = xy0[~np.isnan(xy0).any(axis=1)]
            # 按x从小到大排序
            xy = xy[np.argsort(xy[:,0])]    
            x = xy[:,0]
            y = xy[:,1]
            z1 = np.polyfit(x,y,deg=2)
            p1 =  np.poly1d(z1) #使用次数合成多项式
            y_pre = p1(x)
            r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)

            ## 画密度曲线
            xy_density = np.vstack([x, y])
            z_density = gaussian_kde(xy_density)(xy_density)
            norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())

            range_y = max(y)-min(y)
            plt.ylim(-3*range_y,3*range_y)
            ## jet terrain inferno
            plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
            plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
            plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)')
            plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')

            # plt.title(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS')

def DD_MERSI_VIIRS_plot_part2():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i >9:
            if i in DropVIIRSBand:
                continue
            plt.subplot(2,5,i-9)
            plt.plot(DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)'],DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS'],'+',color = color1)
            ## 画r^2
            xy0 = np.vstack([[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']]])
            xy0 = xy0.transpose()
            # 删除nan值
            xy = xy0[~np.isnan(xy0).any(axis=1)]
            # 按x从小到大排序
            xy = xy[np.argsort(xy[:,0])]    
            x = xy[:,0]
            y = xy[:,1]
            ## 拟合二项式
            try :
                z1 = np.polyfit(x,y,deg=2)
                p1 =  np.poly1d(z1) #使用次数合成多项式
                y_pre = p1(x)
                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                
                ## 画密度曲线
                xy_density = np.vstack([x, y])
                z_density = gaussian_kde(xy_density)(xy_density)
                norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                ## 画图
                range_y = max(y)-min(y)
                plt.ylim(-3*range_y,3*range_y)
                ## jet terrain inferno
                plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)')
                plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')
            except TypeError:
                plt.scatter(0,0,alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)')
                plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')

            # plt.title(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS')

#SRD_MERSI_VIIRS plot ####################################
def SRD_DD_MERSI_VIIRS_plot_part1():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i <=9:
            if i in DropVIIRSBand:
                continue
            plt.subplot(2,5,i+1)
            plt.hist(npsrd[:,i*3+2],color = color1,align='mid',bins=10)
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} SRD_MERSI_VIIRS')
            plt.ylabel('频率')
            # plt.title(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} SRD_MERSI_VIIRS')
    plt.show()

    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i >9:
            if i in DropVIIRSBand:
                continue
            plt.subplot(2,5,i-9)
            try :
                plt.hist(npsrd[:,i*3+2],color = color1,align='mid',bins=10)
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} SRD_MERSI_VIIRS')
                plt.ylabel('频率')
                # plt.title(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} SRD_MERSI_VIIRS')
            except:
                ValueError

    plt.show()

#ORD_MERSI_VIIRS plot ####################################
def ORD_MERSI_VIIRS_plot_part1():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        
        if i <=9:
            if i in DropVIIRSBand:
                continue
            plt.subplot(2,5,i+1)
            plt.hist(npord[:,i*3+2],color = color1,align='mid',bins=10)
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} ORD_MERSI_VIIRS')
            plt.ylabel('频率')
            # plt.title(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} ORD_MERSI_VIIRS')

def ORD_MERSI_VIIRS_plot_part2():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i >9:
            if i in DropVIIRSBand:
                continue
            plt.subplot(2,5,i-9)
            try:
                plt.hist(npord[:,i*3+2],color = color1,align='mid',bins=10)
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} ORD_MERSI_VIIRS')
                plt.ylabel('频率')
                # plt.title(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} ORD_MERSI_VIIRS')
            except:
                ValueError
                
################################################################################################                
################################################################################################
################################################################################################3 #Hist 
def DD_MERSI_MODIS_hist_part1():
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i <=9:
            if i in DropVIIRSBand:
                continue
            x = DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']
            min_value = min(x)
            max_value = max(x)
            range_hist = max_value-min_value
            plt.subplot(2,5,i+1)
            
            plt.hist(DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS'],color = color2,align='mid',bins=25,alpha = 0.7,density=True)
            sns.kdeplot(x,label = '密度图')
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS')
            plt.ylabel('频率')
            
    plt.tight_layout(h_pad=0,w_pad=0.03)    
    plt.show()
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i >9:  
            if i in DropVIIRSBand:
                continue
            plt.subplot(2,5,i-9)
            try :
                x = DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']
                min_value = min(x)
                max_value = max(x)
                range_hist = max_value-min_value
                plt.hist(DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS'],color = color2,align='mid',bins=25,alpha = 0.7,density=True)
                sns.kdeplot(x,label = '密度图')
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS')
                plt.ylabel('频率')
            except:
                ValueError

    plt.tight_layout(h_pad=0,w_pad=0.03)      
    plt.show()
# DD_MERSI_MODIS.to_csv(DD_MERSI_MODISOutFile,index=False,encoding='utf-8-sig')


def DD_Merged_plot_part1():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i <=9:
            ## 不在MODIS中
            if i in DropMODISBand:
                # 不在VIIRS中
                if i in DropVIIRSBand:pass
                # 在VIIRS中
                else:xy0 = np.vstack([[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']]])
            ## 在MODIS中
            else:
                # 不在VIIRS中
                if i  in DropVIIRSBand:
                    xy0 =np.vstack([[DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']]])
                ## 两者都有
                else:
                    x = np.hstack([DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)'],DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)']])
                    y = np.hstack([DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS'],DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']])
                    xy0 = np.vstack([x,y])
            
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
            norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
            range_y = max(y)-min(y)
            plt.ylim(-3*range_y,3*range_y)
                ## jet terrain inferno
            plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                
            plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
            plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 观测值(W/m^2/sr/μm)')
            plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')
            # plt.title(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS')

def DD_Merged_plot_part2():
    for i in range(0,MatchBand_MERSI_VIIRS.shape[0]):
        if i >9:
            if i in DropMODISBand:
                # 不在VIIRS中
                if i in DropVIIRSBand:pass
                # 在VIIRS中
                else:xy0 = np.vstack([[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']]])
            ## 在MODIS中
            else:
                # 不在VIIRS中
                if i  in DropVIIRSBand:
                    xy0 =np.vstack([[DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']]])
                ## 两者都有
                else:
                    x = np.hstack([DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 实际观测值(W/m^2/sr/μm)'],DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)']])
                    y = np.hstack([DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS'],DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']])
                    xy0 = np.vstack([x,y])
                    
                    
            plt.subplot(2,5,i-9)
            ## 画r^2
            xy0 = np.vstack([[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']]])
            xy0 = xy0.transpose()
            # 删除nan值
            xy = xy0[~np.isnan(xy0).any(axis=1)]
            # 按x从小到大排序
            xy = xy[np.argsort(xy[:,0])]    
            x = xy[:,0]
            y = xy[:,1]
            ## 拟合二项式
            try :
                z1 = np.polyfit(x,y,deg=2)
                p1 =  np.poly1d(z1) #使用次数合成多项式
                y_pre = p1(x)
                r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                
                ## 画密度曲线
                xy_density = np.vstack([x, y])
                z_density = gaussian_kde(xy_density)(xy_density)
                norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                ## 画图
                range_y = max(y)-min(y)
                plt.ylim(-3*range_y,3*range_y)
                ## jet terrain inferno
                plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 观测值(W/m^2/sr/μm)')
                plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')
            except TypeError:
                plt.scatter(0,0,alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                plt.xlabel(f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 观测值(W/m^2/sr/μm)')
                plt.title(f'MERSI Band {MatchBand_MERSI_MODIS[i,0]}')
                
def DD_Merged_hist_part1():
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i <=9:
            if i in DropMODISBand:
                # 不在VIIRS中
                if i in DropVIIRSBand:pass
                # 只在VIIRS中
                else: x = DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']
            ## 在MODIS中
            else:
                # 不在VIIRS中
                if i  in DropVIIRSBand:
                    x = DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']
                ## 两者都有
                else:
                    x1 = DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']
                    x2 = DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']
                    x = np.hstack([x1,x2])
            x =  x[~np.isnan(x)]   
            mean_value = np.mean(x)

            plt.subplot(2,5,i+1)
            
            plt.hist(x,color = color2,align='mid',bins=25,alpha = 0.7,density=True)
            plt.axvline(mean_value,color='red',label = '均值')
            sns.kdeplot(x)
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0] } DD',linewidth=1.5)
            # plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS')
            plt.ylabel('频率')
            plt.legend([f'Mean={mean_value:.2f}'],loc='upper right',prop = {'size':10})
            # plt.subplots_adjust(left=1.1, right=1.2)
    plt.tight_layout(h_pad=-0.3,w_pad=-0.3) 
       
def DD_Merged_hist_part2():
    for i in range(0,MatchBand_MERSI_MODIS.shape[0]):
        if i >9:  
            plt.subplot(2,5,i-9)
            if i in DropMODISBand:
                # 不在VIIRS中
                if i in DropVIIRSBand:pass
                # 只在VIIRS中
                else: x = DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']
            ## 在MODIS中
            else:
                # 不在VIIRS中
                if i  in DropVIIRSBand:
                    x = DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']
                ## 两者都有
                else:
                    x1 = DD_MERSI_MODIS[f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS']
                    x2 = DD_MERSI_VIIRS[f'MERSI通道 {MatchBand_MERSI_VIIRS[i,0]} 与 VIIRS通道 {MatchBand_MERSI_VIIRS[i,1]} DD_MERSI_VIIRS']
                    x = np.hstack([x1,x2])
            
            x =  x[~np.isnan(x)]          
            mean_value = np.mean(x)
            plt.hist(x,color = color2,align='mid',bins=25,alpha = 0.7,density=True)
            sns.kdeplot(x)
            plt.axvline(mean_value,color='red',label = '均值',linewidth=1.5)
            # plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0]} 与 MODIS通道 {MatchBand_MERSI_MODIS[i,1]} DD_MERSI_MODIS')
            plt.xlabel(f'MERSI通道 {MatchBand_MERSI_MODIS[i,0] } DD')
            plt.ylabel('频率')
            plt.legend([f'Mean={mean_value:.2f}'],loc='upper right',prop = {'size':10})
            # plt.subplots_adjust(left=0.14, right=0.15)

    plt.tight_layout(h_pad=0,w_pad=0.03)      
    plt.show()               
                
    
if __name__ == '__main__':
    # DD_DD_MERSI_MODIS_plot_part1()
    # DD_MERSI_MODIS_plot_part1()
    DD_Merged_plot_part1()
    plt.show()
    DD_Merged_plot_part2()
    plt.show()
    # DD_Merged_hist_part1()
    # plt.show()
    # DD_Merged_hist_part2()
    # plt.show()
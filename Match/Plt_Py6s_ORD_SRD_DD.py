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
Subject = 'DD'
Month = 'June'

MERSIPy6sFile =r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/Py6s_MERSI_'+Month+'_All_With_MODIS.csv'
MODISPy6sFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/Py6s_MODIS_'+Month+'_All.csv'


DDOutFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/MERSI_MODIS_'+Subject+'.csv'
DD = open(DDOutFile, 'w')
DD.close()
DD = pd.read_csv(DDOutFile,index_col=False,names=['temp'])
NDVImin = 0.05
NDVImax = 0.3
# MatchTablePath= SRDPath
# sOutFile = MatchTablePath.replace('.csv','SS_ForOrigin.csv')

MatchBand = np.array([(1,3),(2,4),(3,1),(4,2),(5,5),(6,6),(7,7),(8,8),(9,9),
             (10,10),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),
             (17,18),(18,19),(19,19)])

dpMODIS = pd.read_csv(MODISPy6sFile,index_col=False)
dpMERSI = pd.read_csv(MERSIPy6sFile,index_col=False)


for k in range(0,MatchBand.shape[0]):

    dpMODIS[f'B{MatchBand[k,1]}s'] = dpMODIS[f'B{MatchBand[k,1]}s'].drop(dpMODIS[(dpMODIS[f'B{MatchBand[k,1]}o']<0.1)].index,axis=0)
    dpMODIS[f'B{MatchBand[k,1]}o'] = dpMODIS[f'B{MatchBand[k,1]}o'].drop(dpMODIS[(dpMODIS[f'B{MatchBand[k,1]}o']<0.1)].index,axis=0)
    dpMERSI[f'B{MatchBand[k,0]}s'] = dpMERSI[f'B{MatchBand[k,0]}s'].drop(dpMODIS[(dpMODIS[f'B{MatchBand[k,1]}o']<0.1)].index,axis=0)
    dpMERSI[f'B{MatchBand[k,0]}o'] = dpMERSI[f'B{MatchBand[k,0]}o'].drop(dpMODIS[(dpMODIS[f'B{MatchBand[k,1]}o']<0.1)].index,axis=0)

dpMERSI.reset_index(drop=True,inplace=True)
dpMODIS.reset_index(drop=True,inplace=True)
SZA = 60
VZA = 60

## 数据清洗
dpMERSI= dpMERSI.drop(dpMERSI[(dpMERSI.NDVI< NDVImin) |(dpMERSI.NDVI> NDVImax) | (dpMERSI.SZA> SZA) | (dpMERSI.VZA> VZA)].index)
dpMODIS = dpMODIS.drop(dpMODIS[(dpMODIS.NDVI< NDVImin)|(dpMODIS.NDVI> NDVImax) | (dpMODIS.SZA> SZA) | (dpMODIS.VZA> VZA)].index)
## 重置索引
dpMERSI.reset_index(drop=True,inplace=True)
dpMODIS.reset_index(drop=True,inplace=True)
# MERSI MODIS SRD +Num
loc = 0
npsrd = np.full([1,MatchBand.shape[0]*3], np.nan)
nptemp = np.zeros((1,MatchBand.shape[0]*3))
nptemp2 = np.zeros((1,MatchBand.shape[0]*3))
npord = np.full([1,MatchBand.shape[0]*3], np.nan)

##################### index 被去掉了！！！！！！！！！！！！！！
## 寻找Num相同，即对应相同点的MERSI和MODIS点
for p,q in tqdm(enumerate(dpMERSI['Num'])):
     for s,t in enumerate(dpMODIS['Num']):
        if q == t :
            for i in range(0,MatchBand.shape[0]):
                    nptemp[loc,i*3 ]  = dpMERSI.loc[p,f'B{MatchBand[i,0]}s']
                    nptemp[loc,i*3+1] = dpMODIS.loc[s,f'B{MatchBand[i,1]}s']
                    nptemp[loc,i*3+2] = dpMERSI.loc[p,f'B{MatchBand[i,0]}s']-dpMODIS.loc[s,f'B{MatchBand[i,1]}s']
                    nptemp2[loc,i*3] = dpMERSI.loc[p,f'B{MatchBand[i,0]}o']
                    nptemp2[loc,i*3+1] = dpMODIS.loc[s,f'B{MatchBand[i,1]}o']
                    nptemp2[loc,i*3+2] = dpMERSI.loc[p,f'B{MatchBand[i,0]}o']-dpMODIS.loc[s,f'B{MatchBand[i,1]}o']
            npsrd = np.concatenate((npsrd,nptemp),axis=0)
            npord = np.concatenate((npord,nptemp2),axis=0)
                

        if t> q:
             break
        else:
             continue
        
# 删除最开始创建的空行
npsrd = np.delete(npsrd,0,axis=0)
npord = np.delete(npord,0,axis=0)
SRD = list(range(0,MatchBand.shape[0]))
ORD = list(range(0,MatchBand.shape[0]))
for i in range(0,MatchBand.shape[0]):
      SRD[i] = npsrd[:,i*3+2]
      ORD[i] = npord[:,i*3+2]
for i in range(0,MatchBand.shape[0]):
        DD.insert(DD.shape[1], f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD' ,ORD[i]-SRD[i])
        DD.insert(DD.shape[1],f'MERSI通道 {MatchBand[i,0]} 实际观测值(W/m^2/sr/μm)',npord[:,i*3])
DD.drop('temp',axis=1,inplace=True) 


#DD plot ####################################
def DD_plot():
    for i in range(0,MatchBand.shape[0]):
        if i <=9:
            plt.subplot(2,5,i+1)
            plt.plot(DD[f'MERSI通道 {MatchBand[i,0]} 实际观测值(W/m^2/sr/μm)'],DD[f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD'],'+',color = color1)


            
            ## 画r^2
            xy0 = np.vstack([[DD[f'MERSI通道 {MatchBand[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD[f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD']]])
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
            # plt.xticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
            # plt.yticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
            # plt.xlim(0,1.2*max(x))
            # plt.ylim(0,1.2*max(y))
            range_y = max(y)-min(y)
            plt.ylim(-3*range_y,3*range_y)
            plt.grid()
            ## jet terrain inferno
            plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
            plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
            plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
            plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 实际观测值(W/m^2/sr/μm)')
            plt.title('MERSI Band '+str(i))
            # plt.title(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD')
    plt.show()

    for i in range(0,MatchBand.shape[0]):
        if i >9:
            plt.subplot(2,5,i-9)
            plt.plot(DD[f'MERSI通道 {MatchBand[i,0]} 实际观测值(W/m^2/sr/μm)'],DD[f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD'],'+',color = color1)


            
            ## 画r^2
            xy0 = np.vstack([[DD[f'MERSI通道 {MatchBand[i,0]} 实际观测值(W/m^2/sr/μm)']],[DD[f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD']]])
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
                plt.grid()
                ## jet terrain inferno
                plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 实际观测值(W/m^2/sr/μm)')
                plt.title('MERSI Band '+str(i))
            except TypeError:
                plt.scatter(0,0,alpha=0.7,marker='+',norm=norm,s=12,linewidth =0.7)
                plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 实际观测值(W/m^2/sr/μm)')
                plt.title('MERSI Band '+str(i))

            # plt.title(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD')

    plt.show()

#SRD plot ####################################
def SRD_plot():
    for i in range(0,MatchBand.shape[0]):
        if i <=9:
            plt.subplot(2,5,i+1)
            plt.hist(npsrd[:,i*3+2],color = color1,align='mid',bins=10)
            plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} SRD')
            plt.ylabel('相对频率')
            # plt.title(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} SRD')
    plt.show()

    for i in range(0,MatchBand.shape[0]):
        if i >9:
            plt.subplot(2,5,i-9)
            try :
                plt.hist(npsrd[:,i*3+2],color = color1,align='mid',bins=10)
                plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} SRD')
                plt.ylabel('相对频率')
                # plt.title(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} SRD')
            except:
                ValueError

    plt.show()

#ORD plot ####################################
def ORD_plot():
    for i in range(0,MatchBand.shape[0]):
        if i <=9:
            plt.subplot(2,5,i+1)
            plt.hist(npord[:,i*3+2],color = color1,align='mid',bins=10)
            plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} ORD')
            plt.ylabel('相对频率')
            # plt.title(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} ORD')
    plt.show()

    for i in range(0,MatchBand.shape[0]):
        if i >9:
            plt.subplot(2,5,i-9)
            try:
                plt.hist(npord[:,i*3+2],color = color1,align='mid',bins=10)
                plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} ORD')
                plt.ylabel('相对频率')
                # plt.title(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} ORD')
            except:
                ValueError

    plt.show()

####################################### #Hist 
def DD_hist():
    for i in range(0,MatchBand.shape[0]):
        if i <=9:
            x = DD[f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD']
            min_value = min(x)
            max_value = max(x)
            range_hist = max_value-min_value
            plt.subplot(2,5,i+1)
            
            plt.hist(DD[f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD'],color = color2,align='mid',bins=25,alpha = 0.7,density=True)
            sns.kdeplot(x,label = '密度图')
            plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD')
            plt.ylabel('相对频率')
            
    plt.tight_layout(h_pad=0,w_pad=0.03)    
    plt.show()
    for i in range(0,MatchBand.shape[0]):
        if i >9:  
            plt.subplot(2,5,i-9)
            try :
                x = DD[f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD']
                min_value = min(x)
                max_value = max(x)
                range_hist = max_value-min_value
                plt.hist(DD[f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD'],color = color2,align='mid',bins=25,alpha = 0.7,density=True)
                sns.kdeplot(x,label = '密度图')
                plt.xlabel(f'MERSI通道 {MatchBand[i,0]} 与 MODIS通道 {MatchBand[i,1]} DD')
                plt.ylabel('相对频率')
            except:
                ValueError

    plt.tight_layout(h_pad=0,w_pad=0.03)      
    plt.show()
# DD.to_csv(DDOutFile,index=False,encoding='utf-8-sig')

if __name__ == '__main__':
    # DD_plot()
    DD_hist()

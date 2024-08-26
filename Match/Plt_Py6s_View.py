import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import math
import pandas as pd
from scipy.stats import gaussian_kde


# MatchTablePath = r"C:/Users/1/Desktop/casestudy/data/MatchRes\LandMatchTable_MERSI_AQUA_CLm^2_5min_202103_Angle_DN30000_202404.csv"
# MatchBand = [(1,3),(2,4),(3,1),(4,2),(6,6),(7,7),(8,8),(9,9),
#              (10,10),(11,12),(12,13),(14,15),(15,16),(16,17),
#              (17,18),(18,19)]

Term = 'MERSI'
Month = 'March'
Land_Sea = 'Land'
NDVImin = 0.05
NDVImax = 0.3
Angle = 60
match Term:
        case 'MERSI' :
                Py6sFile = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/Py6s_'+Term+'_'+Month+'_All.csv'
                # Py6sFile = r"C:\Users\1\Desktop\casestudy\data\MatchRes\Py6sMERSI_Conti_All.csv"
        case 'MODIS' :
                if Land_Sea == 'Land':
                        Py6sFile = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/Py6s_'+Term+'_'+Month+'_All.csv'
                # if Land_Sea == 'Sea':
                #         Py6sFile = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/Py6s_'+Term+'_'+Month+'_All.csv'
        case 'VIIRS' :
                Py6sFile = r'C:/Users/1/Desktop/casestudy/data/MatchRes/'+Month+'/Py6s_'+Term+'_'+Month+'_All.csv'


sOutFile = Py6sFile.replace('.csv','ForOrigin.csv')
#创建空csv文件
fp = open(sOutFile, 'w')
fp.close()
#读取py6s结果
dp = pd.read_csv(Py6sFile,index_col=False)

# MERSI
df = pd.read_csv(sOutFile,index_col = False,names=['MERSI 通道1实际观测值(W/m^2/sr/μm)','MERSI 通道1模拟观测值(W/m^2/sr/μm)','MERSI 通道2实际观测值(W/m^2/sr/μm)','MERSI 通道2模拟观测值(W/m^2/sr/μm)','MERSI 通道3实际观测值(W/m^2/sr/μm)','MERSI 通道3模拟观测值(W/m^2/sr/μm)','MERSI 通道4实际观测值(W/m^2/sr/μm)','MERSI 通道4模拟观测值(W/m^2/sr/μm)','MERSI 通道5实际观测值(W/m^2/sr/μm)','MERSI 通道5模拟观测值(W/m^2/sr/μm)',
                                                   'MERSI 通道6实际观测值(W/m^2/sr/μm)','MERSI 通道6模拟观测值(W/m^2/sr/μm)','MERSI 通道7实际观测值(W/m^2/sr/μm)','MERSI 通道7模拟观测值(W/m^2/sr/μm)','MERSI 通道8实际观测值(W/m^2/sr/μm)','MERSI 通道8模拟观测值(W/m^2/sr/μm)','MERSI 通道9实际观测值(W/m^2/sr/μm)','MERSI 通道9模拟观测值(W/m^2/sr/μm)','MERSI 通道10实际观测值(W/m^2/sr/μm)','MERSI 通道10模拟观测值(W/m^2/sr/μm)',
                                                   'MERSI 通道11实际观测值(W/m^2/sr/μm)','MERSI 通道11模拟观测值(W/m^2/sr/μm)','MERSI 通道12实际观测值(W/m^2/sr/μm)','MERSI 通道12模拟观测值(W/m^2/sr/μm)','MERSI 通道13实际观测值(W/m^2/sr/μm)','MERSI 通道13模拟观测值(W/m^2/sr/μm)','MERSI 通道14实际观测值(W/m^2/sr/μm)','MERSI 通道14模拟观测值(W/m^2/sr/μm)','MERSI 通道15实际观测值(W/m^2/sr/μm)',
                                                   'MERSI 通道15模拟观测值(W/m^2/sr/μm)','MERSI 通道16实际观测值(W/m^2/sr/μm)','MERSI 通道16模拟观测值(W/m^2/sr/μm)','MERSI 通道17实际观测值(W/m^2/sr/μm)','MERSI 通道17模拟观测值(W/m^2/sr/μm)','MERSI 通道18实际观测值(W/m^2/sr/μm)','MERSI 通道18模拟观测值(W/m^2/sr/μm)','MERSI 通道19实际观测值(W/m^2/sr/μm)','MERSI 通道19模拟观测值(W/m^2/sr/μm)'])

# MODIS
dm = pd.read_csv(sOutFile,index_col = False,names=['MODIS 通道1实际观测值(W/m^2/sr/μm)','MODIS 通道1模拟观测值(W/m^2/sr/μm)','MODIS 通道2实际观测值(W/m^2/sr/μm)','MODIS 通道2模拟观测值(W/m^2/sr/μm)','MODIS 通道3实际观测值(W/m^2/sr/μm)','MODIS 通道3模拟观测值(W/m^2/sr/μm)','MODIS 通道4实际观测值(W/m^2/sr/μm)','MODIS 通道4模拟观测值(W/m^2/sr/μm)','MODIS 通道5实际观测值(W/m^2/sr/μm)','MODIS 通道5模拟观测值(W/m^2/sr/μm)',
                                                   'MODIS 通道6实际观测值(W/m^2/sr/μm)','MODIS 通道6模拟观测值(W/m^2/sr/μm)','MODIS 通道7实际观测值(W/m^2/sr/μm)','MODIS 通道7模拟观测值(W/m^2/sr/μm)','MODIS 通道8实际观测值(W/m^2/sr/μm)','MODIS 通道8模拟观测值(W/m^2/sr/μm)','MODIS 通道9实际观测值(W/m^2/sr/μm)','MODIS 通道9模拟观测值(W/m^2/sr/μm)','MODIS 通道10实际观测值(W/m^2/sr/μm)','MODIS 通道10模拟观测值(W/m^2/sr/μm)',
                                                   'MODIS 通道11实际观测值(W/m^2/sr/μm)','MODIS 通道11模拟观测值(W/m^2/sr/μm)','MODIS 通道12实际观测值(W/m^2/sr/μm)','MODIS 通道12模拟观测值(W/m^2/sr/μm)','MODIS 通道13实际观测值(W/m^2/sr/μm)','MODIS 通道13模拟观测值(W/m^2/sr/μm)','MODIS 通道14实际观测值(W/m^2/sr/μm)','MODIS 通道14模拟观测值(W/m^2/sr/μm)','MODIS 通道15实际观测值(W/m^2/sr/μm)',
                                                   'MODIS 通道15模拟观测值(W/m^2/sr/μm)','MODIS 通道16实际观测值(W/m^2/sr/μm)','MODIS 通道16模拟观测值(W/m^2/sr/μm)','MODIS 通道17实际观测值(W/m^2/sr/μm)','MODIS 通道17模拟观测值(W/m^2/sr/μm)','MODIS 通道18实际观测值(W/m^2/sr/μm)','MODIS 通道18模拟观测值(W/m^2/sr/μm)','MODIS 通道19实际观测值(W/m^2/sr/μm)','MODIS 通道19模拟观测值(W/m^2/sr/μm)'])

dv = pd.read_csv(sOutFile,index_col = False,names=['VIIRS 通道1实际观测值(W/m^2/sr/μm)','VIIRS 通道1模拟观测值(W/m^2/sr/μm)','VIIRS 通道2实际观测值(W/m^2/sr/μm)','VIIRS 通道2模拟观测值(W/m^2/sr/μm)','VIIRS 通道3实际观测值(W/m^2/sr/μm)','VIIRS 通道3模拟观测值(W/m^2/sr/μm)','VIIRS 通道4实际观测值(W/m^2/sr/μm)','VIIRS 通道4模拟观测值(W/m^2/sr/μm)','VIIRS 通道5实际观测值(W/m^2/sr/μm)','VIIRS 通道5模拟观测值(W/m^2/sr/μm)',
                                                   'VIIRS 通道6实际观测值(W/m^2/sr/μm)','VIIRS 通道6模拟观测值(W/m^2/sr/μm)','VIIRS 通道7实际观测值(W/m^2/sr/μm)','VIIRS 通道7模拟观测值(W/m^2/sr/μm)','VIIRS 通道8实际观测值(W/m^2/sr/μm)','VIIRS 通道8模拟观测值(W/m^2/sr/μm)','VIIRS 通道9实际观测值(W/m^2/sr/μm)','VIIRS 通道9模拟观测值(W/m^2/sr/μm)','VIIRS 通道10实际观测值(W/m^2/sr/μm)','VIIRS 通道10模拟观测值(W/m^2/sr/μm)',
                                                   'VIIRS 通道11实际观测值(W/m^2/sr/μm)'])


# df.columns = ['B1o','B1s','B2o','B2s','B3o','B3s','B4o','B4s','B5o','B5s','B6o','B6s','B7o','B7s','B8o','B8s','B9o','B9s','B10o','B10s','B11o','B11s','B12o','B12s','B13o','B13s','B14o','B14s','B15o','B15s','B16o','B16s','B17o','B17s','B18o','B18s','B19o','B19s']

## 筛选条件
# 删除符合条件的指定行，并替换原始df
# DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors=‘raise’)
# 1、labels：要删除的标签，一个或者多个(以list形式)；
# 2、axis：指定哪一个轴，=0删除行，=1删除列；
# 3、columns：指定某一列或者多列(以list形式)；
# 4、level：索引等级，针对多重索引的情况；
# 5、inplaces：是否替换原来的dataframe，=True代表直接替换原始df，=False代表不替换原始df


match Term:
        case 'MERSI' :
                do = dp.drop(dp[(dp.NDVI< NDVImin)|(dp.NDVI> NDVImax) | (dp.SZA> Angle) | (dp.VZA> Angle)].index)
        case 'MODIS' :
                if Land_Sea == 'Sea' :
                        do=dp
                if Land_Sea == 'Land' :
                        do = dp.drop(dp[(dp.NDVI< NDVImin)|(dp.NDVI> NDVImax) | (dp.SZA> Angle) | (dp.VZA> Angle)].index)
        case 'VIIRS' :
                do = dp.drop(dp[(dp.NDVI< NDVImin)|(dp.NDVI> NDVImax) | (dp.SZA> Angle) | (dp.VZA> Angle)].index)
# do = dp


#MODIS
match Term:
        case 'MERSI' :
                for i in range(1,20):

                        df[f'MERSI 通道{i}实际观测值(W/m^2/sr/μm)'] = do[f'B{i}o'].drop(do[(do[f'B{i}o']<0.1)].index,axis=0)
                        df[f'MERSI 通道{i}模拟观测值(W/m^2/sr/μm)'] = do[f'B{i}s'].drop(do[(do[f'B{i}o']<0.1)].index,axis=0)
        case 'MODIS' :
                for i in range(1,20):
        
                        dm[f'MODIS 通道{i}实际观测值(W/m^2/sr/μm)'] = do[f'B{i}o'].drop(do[(do[f'B{i}o']<0.1)].index,axis=0)
                        dm[f'MODIS 通道{i}模拟观测值(W/m^2/sr/μm)'] = do[f'B{i}s'].drop(do[(do[f'B{i}o']<0.1)].index,axis=0)
        case 'VIIRS' :
                for i in range(1,11):
                        
                        dv[f'VIIRS 通道{i}实际观测值(W/m^2/sr/μm)'] = do[f'B{i}o'].drop(do[(do[f'B{i}o']<0.01)|(do[f'B{i}o']>=np.nan)].index,axis=0)
                        dv[f'VIIRS 通道{i}模拟观测值(W/m^2/sr/μm)'] = do[f'B{i}s'].drop(do[(do[f'B{i}o']<0.01)|((do[f'B{i}o']>=np.nan))].index,axis=0)

# MODIS
# df = df.drop('B5o',axis=1)
# df = df.drop('B5s',axis=1)
# df = df.drop('B11o',axis=1)
# df = df.drop('B11s',axis=1)
# df = df.drop('B14o',axis=1)
# df = df.drop('B14s',axis=1)

# MERSI
# df = df.drop('MERSI 通道5实际观测值(W/m^2/sr/μm)',axis=1)
# df = df.drop('MERSI 通道5模拟观测值(W/m^2/sr/μm)',axis=1)
# df = df.drop('MERSI 通道13实际观测值(W/m^2/sr/μm)',axis=1)
# df = df.drop('MERSI 通道13模拟观测值(W/m^2/sr/μm)',axis=1)
# df = df.drop('MERSI 通道19实际观测值(W/m^2/sr/μm)',axis=1)
# df = df.drop('MERSI 通道19模拟观测值(W/m^2/sr/μm)',axis=1)

# # 
# dm = dm.drop('MODIS 通道5实际观测值(W/m^2/sr/μm)',axis=1)
# dm = dm.drop('MODIS 通道5模拟观测值(W/m^2/sr/μm)',axis=1)
# dm = dm.drop('MODIS 通道11实际观测值(W/m^2/sr/μm)',axis=1)
# dm = dm.drop('MODIS 通道11模拟观测值(W/m^2/sr/μm)',axis=1)
# dm = dm.drop('MODIS 通道14实际观测值(W/m^2/sr/μm)',axis=1)
# dm = dm.drop('MODIS 通道14模拟观测值(W/m^2/sr/μm)',axis=1)

# df.to_csv(sOutFile,index=False)
match Term:
        case 'MERSI':
                k = 0
                for j in range(1,20):  
                        if j <= 10:
                                        k+=1
                                        xy0 = np.vstack([[df[f'MERSI 通道{j}实际观测值(W/m^2/sr/μm)']],[df[f'MERSI 通道{j}模拟观测值(W/m^2/sr/μm)']]])
                                        xy0 = xy0.transpose()
                                        xy = xy0[~np.isnan(xy0).any(axis=1)]
                                        # 删除nan值
                                        xy = xy[np.argsort(xy[:,0])]    
                                        # 按x从小到大排序
                                        x = xy[:,0]
                                        y = xy[:,1]
                                        z1 = np.polyfit(x,y,deg=2)
                                        p1 = np.poly1d(z1) #使用次数合成多项式
                                        y_pre = p1(x)
                                        r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                        xy_density = np.vstack([x, y])
                                        z_density = gaussian_kde(xy_density)(xy_density)
                                        norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                                        plt.subplot(2,5,k)

                                        if j ==5 or j ==7:
                                                plt.xticks(np.arange(0,1.2*math.ceil(0.1*max(x))*10+1,0.2*math.ceil(0.1*max(x))*10))
                                                plt.yticks(np.arange(0,1.2*math.ceil(0.1*max(x))*10+1,0.2*math.ceil(0.1*max(x))*10))
                                                # plt.xlim(0,math.ceil(0.1*max(x))*10)
                                                # plt.ylim(0,math.ceil(0.1*max(x))*10)
                                        else:
                                                plt.xticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                                                plt.yticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                                        plt.xlim(0,1.2*max(x))
                                        plt.ylim(0,1.2*max(y))
                                        plt.grid()
                                        ## jet terrain inferno
                                        plt.scatter(x,y,c=z_density,cmap='jet',alpha=0.7,marker='+',norm=norm,s=12,linewidth =1.0)
                                        plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                                        plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                                        # plt.axis('equal')
                                        plt.title('MERSI Band '+str(j))
                                        
                plt.show()
                k = 0
                for j in range(1,20): 
                                if j > 10:
                                        k+=1
                                        xy0 = np.vstack([[df[f'MERSI 通道{j}实际观测值(W/m^2/sr/μm)']],[df[f'MERSI 通道{j}模拟观测值(W/m^2/sr/μm)']]])
                                        xy0 = xy0.transpose()
                                        # 删除nan值
                                        xy = xy0[~np.isnan(xy0).any(axis=1)]   
                                        # 按x从小到大排序    
                                        xy = xy[np.argsort(xy[:,0])]                      
                                        x = xy[:,0]
                                        y = xy[:,1]
                                        z1 = np.polyfit(x,y,deg=2)
                                        p1 = np.poly1d(z1) #使用次数合成多项式
                                        y_pre = p1(x)
                                        r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                        xy_density = np.vstack([x, y])
                                        z_density = gaussian_kde(xy_density)(xy_density)
                                        norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                                        plt.subplot(2,5,k)
                                        plt.xticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                                        plt.yticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                                        plt.xlim(0,1.2*max(x))
                                        plt.ylim(0,1.2*max(y))
                                        plt.grid()
                                        ## jet terrain inferno
                                        plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =1.0)
                                        plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                                        plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                                        # plt.axis('equal')
                                        plt.title('MERSI Band '+str(j))
                                
                                
                                       
                plt.show()
                df.to_csv(sOutFile,index=False,encoding='utf-8-sig')
        case 'MODIS':
                k = 0
                for j in range(1,20): 
                                
                                if j <= 10:
                                        k+=1
                                        xy0 = np.vstack([[dm[f'MODIS 通道{j}实际观测值(W/m^2/sr/μm)']],[dm[f'MODIS 通道{j}模拟观测值(W/m^2/sr/μm)']]])
                                        xy0 = xy0.transpose()
                                        xy = xy0[~np.isnan(xy0).any(axis=1)]
                                        # 删除nan值
                                        xy = xy[np.argsort(xy[:,0])]    
                                        # 按x从小到大排序
                                        x = xy[:,0]
                                        y = xy[:,1]
                                        z1 = np.polyfit(x,y,2)
                                        p1 = np.poly1d(z1) #使用次数合成多项式
                                        y_pre = p1(x)
                                        r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                        xy_density = np.vstack([x, y])
                                        z_density = gaussian_kde(xy_density)(xy_density)
                                        norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                                        plt.subplot(2,5,k)
                                        plt.xticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                                        plt.yticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                                        plt.xlim(0,1.2*max(x))
                                        plt.ylim(0,1.2*max(y))
                                        plt.grid()
                                        ## jet terrain inferno
                                        plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =1.0)
                                        plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                                        plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                                        # plt.axis('equal')
                                        plt.title('MODIS Band '+str(j))
                                
                                        
                plt.show()
                k = 0
                for j in range(1,20): 
                                
                                if j > 10:
                                        k+=1
                                        if j == 15: 
                                                plt.subplot(2,5,k)
                                                plt.scatter(0,0,color='blue',alpha=0.1)
                                                # plt.plot(x,y_pre,color='red',label='Fitting Curve')
                                                plt.title('MODIS Band '+str(j))
                                                continue
                                                
                                        if j == 16: 
                                                plt.subplot(2,5,k)
                                                plt.scatter(0,0,color='blue',alpha=0.1)
                                                # plt.plot(x,y_pre,color='red',label='Fitting Curve')
                                                plt.title('MODIS Band '+str(i))
                                                continue
                                        xy0 = np.vstack([[dm[f'MODIS 通道{j}实际观测值(W/m^2/sr/μm)']],[dm[f'MODIS 通道{j}模拟观测值(W/m^2/sr/μm)']]])
                                        xy0 = xy0.transpose()
                                        # 删除nan值
                                        xy = xy0[~np.isnan(xy0).any(axis=1)]   
                                        # 按x从小到大排序    
                                        xy = xy[np.argsort(xy[:,0])]                      
                                        x = xy[:,0]
                                        y = xy[:,1]
                                        z1 = np.polyfit(x,y,2)
                                        p1 = np.poly1d(z1) #使用次数合成多项式
                                        y_pre = p1(x)
                                        r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                                        xy_density = np.vstack([x, y])
                                        z_density = gaussian_kde(xy_density)(xy_density)
                                        norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                                        plt.subplot(2,5,k)
                                        plt.xticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                                        plt.yticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                                        plt.xlim(0,1.2*max(x))
                                        plt.ylim(0,1.2*max(y))
                                        plt.grid()
                                        ## jet terrain inferno
                                        plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =1.0)
                                        plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                                        plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                                        # plt.axis('equal')
                                        plt.title('MERSI Band '+str(j))
                                
                                
                                       
                plt.show()
                dm.to_csv(sOutFile,index=False,encoding='utf-8-sig')
        case 'VIIRS':
                for i in range(1,11):
                        xy0 = np.vstack([[dv[f'VIIRS 通道{i}实际观测值(W/m^2/sr/μm)']],[dv[f'VIIRS 通道{i}模拟观测值(W/m^2/sr/μm)']]])
                        xy0 = xy0.transpose()
                        xy = xy0[~np.isnan(xy0).any(axis=1)]
                        x = xy[:,0]
                        y = xy[:,1]
                        z1 = np.polyfit(x,y,1)
                        p1 = np.poly1d(z1) #使用次数合成多项式
                        y_pre = p1(x)
                        r2 = 1 - np.sum((y - y_pre)**2) / np.sum((y - np.mean(y))**2)
                        xy_density = np.vstack([x, y])
                        z_density = gaussian_kde(xy_density)(xy_density)
                        norm = mpl.colors.Normalize(vmin=z_density.min(), vmax=z_density.max())
                        plt.subplot(3,4,i)
                        plt.xticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                        plt.yticks(np.arange(0,1.2*math.ceil(0.02*max(x))*50,0.2*math.ceil(0.02*max(x))*50))
                        plt.xlim(0,1.2*max(x))
                        plt.ylim(0,1.2*max(y))
                        plt.grid()
                        ## jet terrain inferno
                        plt.scatter(x,y,c=z_density,cmap='plasma',alpha=0.7,marker='+',norm=norm,s=12,linewidth =1.0)
                        plt.plot(x,y_pre,color='red',label='Fitting Curve',alpha=0.5)
                        plt.legend([f'{len(x)} Point',f'r^2={r2:.3f}'],loc='upper left',prop = {'size':8})
                        # plt.axis('equal')
                        plt.title('VIIRS Band '+str(i))
                plt.show()
                # dv.to_csv(sOutFile,index=False,encoding='utf-8-sig')











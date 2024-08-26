# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
import datetime
import pandas as pd
# nc=Dataset(r'C:\Users\1\Desktop\casestudy\era5\adaptor.mars.internal-1698004431.3961456-17920-14-ee9a0a20-681b-408d-878e-35d75056e7a0.nc')
nc=Dataset(r"D:\ERA5\atmos\20210322.nc")
print(nc.variables.keys())

long= nc.variables['longitude'][:]
lati= nc.variables['latitude'][:]
for var in nc.variables.keys():
    data=nc.variables[var][:].data
    print(var,data.shape)


print(long[0],long[-1],lati[0],lati[-1])
print(long.shape,lati.shape)

# #  提取mean数据
long1 = 16
lati1 = 24
# print(long[long1],lati[lati1])
Pressure =nc.variables['level'][:].data
Pressure = Pressure[0:34]
# # 取全部数据
O3 = nc.variables['o3'][:].data
Tem = nc.variables['t'][:].data
Rhumd = nc.variables['r'][:].data
geo = nc.variables['z'][:].data
Shum = nc.variables['q'][:].data
# 取34层数据
X = [O3,Tem,Rhumd,geo,Shum]
#

for i in range(5):
    X[i] = X[i][1,0:34,long1,lati1]
# 臭氧单位转换Kg/Kg to g/m3
X[0] = X[0]*(Pressure*100)/(287.058*X[1])*1000
# 水气单位转换Kg/Kg to g/m3
X[4] = X[4]*(Pressure*100)
X[4] = X[4]/(287.058*X[1])*1000
# 位势高度转换km
X[3] = X[3]/9.8/1000
# 气压帕斯卡转为毫巴
Pressure = Pressure



Y = [X[3],Pressure,X[1],X[0],X[4]]


data = {
    "altitude": Y[0],
    "pressure": Y[1],
    "temp":Y[2],
    "water":Y[4],
    "ozone":Y[3]
}


df = pd.DataFrame(data)
# 倒置
df = df.iloc[::-1]
df.columns = ["altitude", "pressure","temperature", "water","ozone"]


df.to_csv(r'C:\Users\1\Desktop\casestudy\1016-reverse.csv',index=False,sep=',')



    #plt.savefig('Rainf_0.png',dpi=300)

# <matplotlib.colorbar.Colorbar at 0x2b48e65ebe0>
# time=nc.variables['time'][:].data
# data=nc.variables['t2m'][:]
# print(data.shape)
#该文件是辐射资料，来自ECMWF网站
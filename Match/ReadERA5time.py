# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
import datetime
import pandas as pd
path = r"C:\Users\1\Downloads\adaptor.mars.internal-1720604606.1148098-5106-5-2fc4454d-ec22-4fcd-854a-dd4b0ee5726a.nc"
nc=Dataset(path)
# sListFile = r'C:\Users\1\Desktop\py\Match\listAero.dat'
# fp = open(sListFile, 'r')
# filelist = [x.rstrip() for x in fp]
# fp.close()
print(nc.variables.keys())

lon = nc.variables['longitude'][:]
lat = nc.variables['latitude'][:]
# nc = Dataset(filelist[0],'r')
print(lon[0],lon[-1],lat[0],lat[-1])
# print(nc.variables.keys())

t=np.array(nc.variables['time'][2])

#时间转化
st=datetime.datetime(1900,1,1,0,0)
a=st+datetime.timedelta(hours=int(t))
                        
print(a)
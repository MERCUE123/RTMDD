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
import math

# sListFile = r'C:\Users\1\Desktop\py\Match\listAero.dat'
MODISAeroPath = r'D:\ERA5\Aero\MODIS_Aero' #MODIS AERO FILE
AeroFiles = glob.glob(MODISAeroPath + '/*.nc')
Aerofile = []


nc = Dataset(AeroFiles[0],'r')
print(nc.variables.keys())
A = nc.variables['Aerosol_Optical_Thickness_550_Land_Count'][:]
B = nc.variables['Aerosol_Optical_Thickness_550_Land_Mean'][:]
plt.imshow(B)
plt.show()
Lati = nc.variables['Latitude'][:]
Lont = nc.variables['Longitude'][:]
plt.imshow(Lont)
plt.show()
# long= nc.variables['longitude'][:]
# lon = np.array(long)
# lati= nc.variables['latitude'][:]
# lat = np.array(lati)

# AeroKeyword = 'NOAA20.A{}{}'.format(str(ERA5time.year), str(days).zfill(3))
# AeroFiles = glob.glob(MODISAeroPath + '/*.nc')
# Aerofile = []
# for k in AeroFiles:
#             # 确定日期
#     if AeroKeyword in os.path.basename(k):
#                 # print('Find Match ' + time4Y2M2D)  # 绝对路径
#         Aerofile = k
#     break
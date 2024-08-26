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
deg = 180.0/np.pi
# sListFile = r'C:\Users\1\Desktop\py\Match\listAero.dat'
# sListFile = r"C:\Users\1\Desktop\py\Match\listAero_fine_mode.dat"
uFile = r"D:\ERA5\Wind\March\uWind.nc"
vFile = r"D:\ERA5\Wind\March\vWind.nc"

u = Dataset(uFile,'r')
v = Dataset(vFile,'r')
print(u.variables.keys())
long= u.variables['longitude'][:]
lon = np.array(long)
lati= u.variables['latitude'][:]
lat = np.array(lati)
print(long[0],long[-1],lati[0],lati[-1])
Timetable = [2,5,8,10,13,16,18,21,24,29]
Day = 2
Hour = 5
i = 10
j = 10
Dayindex = Timetable.index(Day)
Hourindex = Timetable.index(Hour)
uWind = u.variables['u'][(Day-1)*24+Hour][0,i,j]
print(u.variables['u'].shape)
vWind = v.variables['v'][(Day-1)*24+Hour][0,i,j]

WindSpeed = math.sqrt(uWind**2+vWind**2)
wdir1 =  180.0 + np.arctan2(uWind, vWind)*deg
wdir2 =  270.0 - np.arctan2( vWind, uWind)*deg
print(wdir1,wdir2)


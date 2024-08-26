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
# sListFile = r"C:\Users\1\Desktop\py\Match\listAero_fine_mode.dat"
sListFile = r'D:\ERA5\Chla\March'
filelist = glob.glob(sListFile + '/*.nc')

nc = Dataset(filelist[0],'r')
print(nc.variables.keys())
chla = nc.variables['chlor_a'][:]
print(chla.shape)
long= nc.variables['lon'][:]
lon = np.array(long)
lati= nc.variables['lat'][:]
lat = np.array(lati)
print(chla[0,100,100])
chla.shape = 4320,8640
plt.imshow(chla)
plt.show()
print(long[0],long[-1],lati[0],lati[-1])

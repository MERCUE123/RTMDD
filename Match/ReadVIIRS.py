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
import pyhdf.SD as HDF
import h5py
import time
mtime0 = datetime(1993, 1, 1, 0, 0)
time0 = datetime(2021, 1, 1, 0, 0)
# sListFile = r'C:\Users\1\Desktop\py\Match\listVIIRS.dat'
# VIIRSPath = r'D:\ERA5\VIIRS\MODIS_VIIRS' #MODIS AERO FILE
# VIIRSFiles = glob.glob(MODISVIIRSPath + '/*.nc')
VIIRSfile = r"D:\VIIRS\June\1KM&GEO&CLM\VJ102MOD.A2021061.0054.021.2021072215243.nc"
VJ103file = r'D:\VIIRS\June\1KM&GEO&CLM\VJ103MOD.A2021062.0924.021.2021072221325.nc'
# netcdf4 用不了 靠！！
VJ102File = h5py.File(VIIRSfile)
VJ103File= h5py.File(VJ103file)
print(VJ102File.keys())
# print(nc.__dict__.keys())
M01= VJ102File['observation_data']['M01']
M02= VJ102File['observation_data']['M02']

StartTimestr = str(VJ103File.attrs['StartTime'])
EndTime = str(VJ103File.attrs['EndTime'])
StartTimestr = StartTimestr[2:21]
EndTime = EndTime[2:21]
# StartTimestr2 = StartTimestr[13:21]
StartTimeDate = datetime.strptime(StartTimestr,  "%Y-%m-%d %H:%M:%S")
StartSeconds = (StartTimeDate-time0).total_seconds()
EndTimeDate = datetime.strptime(EndTime,  "%Y-%m-%d %H:%M:%S")
EndTimeSeconds = (EndTimeDate-time0).total_seconds()

scale= VJ102File['observation_data']['M02'].attrs['scale_factor']
# linescantime = (EndTime -StartTime) /1000

end =1

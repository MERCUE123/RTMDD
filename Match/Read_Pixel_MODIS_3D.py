import sys
import numpy as np
import math
from datetime import datetime, timedelta
import scipy.constants as const
from netCDF4 import Dataset
import os, fnmatch
import glob

MODIS_File = r"D:\MODIS\SampleData\MYD03.A2021061.1750.061.2021062170543_0.25X0.25.dat"
FY3D_File = r"D:\FY3D\SampleData\FY3D_MERSI_GBAL_L1_20210302_1745_GEO1K_MS_0.25X0.25.dat"

NMODIS = 28
NFY3D = 24
row = 721
col = 1440
MODIS = np.fromfile(MODIS_File,dtype=np.float32)
MODIS.shape = 28,row,col
FY3D = np.fromfile(FY3D_File,dtype=np.float32)
FY3D.shape = 24,row,col

nrow =120
ncol = 1158
Band = 4
Pix_MODIS = MODIS[Band,nrow,ncol]
Pix_FY3D = FY3D[Band,nrow,ncol]

print('MODIS'+'Band'+str(Band)+' value='+str(Pix_MODIS))
print('FY3D'+'Band'+str(Band)+' value='+str(Pix_FY3D))
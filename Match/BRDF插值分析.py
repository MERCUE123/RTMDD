
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
Month = 'June'
MODIS = list(np.array([0.645,0.856,0.466,0.554,1.241,1.628,2.113,0.412,0.442,0.487,
                    0.530,0.547,0.666,0.678,0.747,0.867,0.904,0.936,0.935]))

MERSI = list(np.array([0.471,0.554,0.653,0.868,1.381,1.645,2.125,0.411,0.444,0.490,
                    0.556,0.670,0.709,0.746,0.865,0.905,0.936,0.940,1.030])) 
  
MODISdic = {'Band {}'.format(i+1): value for i, value in enumerate(MODIS)}

BRDFpath = r'C:/Users/1/Desktop/casestudy/data/MODIS/'+Month+'/BRDF/BRDFC1_Sampleto_0.25x0.25MODIS'

BRDFCH19Path = r'C:/Users/1/Desktop/casestudy/data/MODIS/'+Month+'/BRDF/BRDFC1_Sampleto_19CH_MERSI'
BRDFCH19Path2 = r'C:/Users/1/Desktop/casestudy/data/MODIS/'+Month+'/BRDF/BRDFC1_Sampleto_19CH_MODIS'

BRDFfiles = glob.glob(BRDFpath + '/*.dat')
BRDF19files = glob.glob(BRDFCH19Path + '/*.dat')
BRDF19files2 = glob.glob(BRDFCH19Path2 + '/*.dat')
            # plt.subplot(3, 1, 1)
            # plt.plot( x, y1, 'ro', Aqua,Ep1, 'b*' )
            # plt.subplot(3, 1, 2)
            # plt.plot(x, y2, 'ro', Aqua, Ep2, 'b*')
            # plt.subplot(3, 1, 3)
            # plt.plot(x, y3, 'ro', Aqua, Ep3, 'b*')
            
            # plt.show()
BRDF7 = np.fromfile(BRDFfiles[12],dtype=np.float32)
BRDF7.shape = 21,721,1440
# BRDF19 = np.fromfile(BRDF19files2[12], dtype=np.float32)
# BRDF19.shape = 57, 721, 1440
BRDF19_MERSI = np.fromfile(BRDF19files[12], dtype=np.float32)
BRDF19_MERSI.shape = 57, 721, 1440
band1_7 = np.zeros(7)
band1_19 = np.zeros(19)
band2_19 = np.zeros(19)
lat = 352
lon = 101
## 0 for fiso 1 for fgeo 2 for fvol
k = 1
for i in range(7):
    
    band1_7[i] = BRDF7[i*3+k,lat,lon]
for i in range(19):
    # band1_19[i] = BRDF19[i+1,lat,lon]
    band2_19[i] = BRDF19_MERSI[19*k+i,lat,lon]

plt.scatter( MODIS[0:7], band1_7, color='teal'  ,label = 'origin',marker='o',alpha=1)
# plt.scatter(MODIS,band1_19, color = 'tomato',label = 'inter',marker='+',alpha=1)
plt.scatter(MERSI,band2_19, color = 'tomato',label = 'inter',marker='+',alpha=1)
plt.xlabel('wavelength(um)')
plt.ylabel('Band1_fiso')
plt.legend(loc=2)
plt.show()

# plt.scatter(MODIS[0:7], band1_7, color='teal'  ,label = 'origin',marker='o',alpha=1)

# plt.xlabel('wavelength(um)')
# plt.ylabel('Band1_fiso')
# plt.legend(loc=2)
# plt.show()
# for k,BRDFfile in enumerate(BRDFfiles):
#     # if k==0 :continue
#     BRDF7 = np.fromfile(BRDFfile, dtype=np.float32)
#     BRDF7.shape = 21, 721, 1440

#     BRDF19 = np.fromfile(BRDF19files[k], dtype=np.float32)
#     BRDF19.shape = 57, 721, 1440

#     BRDF19_2 = np.fromfile(BRDF19files2[k], dtype=np.float32)
#     BRDF19_2.shape = 57, 721, 1440
    
#     D = BRDF19[0,:,:] - BRDF19_2[2,:,:]
#     a = BRDF7[6,63,76]
#     b = BRDF19[0,63,76]
#     plt.imshow(D)
#     plt.show()

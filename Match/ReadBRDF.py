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

BRDFpath = r'D:\MODIS\BRDF\BRDF_Sampleto_19CH_MODIS'
BRDF25path = r'D:\MODIS\BRDF\BRDFsample_0.25x0.25MODIS'
BRDFfiles = glob.glob(BRDFpath + '/*.dat')
BRDF25files = glob.glob(BRDF25path + '/*.dat')



BRDF = np.fromfile(BRDFfiles[5], dtype=np.float32)
BRDF.shape = 57, 721, 1440
BRDF25 = np.fromfile(BRDF25files[5], dtype=np.float32)
BRDF25.shape = 21, 721, 1440

Diff = BRDF[2,:,:] - BRDF25[6,:,:]  

plt.imshow(Diff)
plt.show()
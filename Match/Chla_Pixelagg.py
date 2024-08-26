import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import glob
import matplotlib.pyplot as plt
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

nRow1 = 4320
nCol1 = 8640
nRow2 = 721
nCol2 = 1440
lon0 = 0.0
lat0 = 90.0
dxy = 0.25
suffix = '_Chla_0.25X0.25.dat'
fillvalue1 = 0
fillvalue2 =-0
Chlapath = r'D:\ERA5\Chla\March'
# Outpath = r'D:\ERA5\Chla\June\MODISChla_Sampleto_0.25?x0.25'
OUT = np.zeros((nRow2, nCol2), dtype=np.float32)
COUNT = np.zeros((nRow2, nCol2), dtype=np.float32)
COUNTFILL = np.zeros((nRow2, nCol2), dtype=np.float32)

def RSLibImport(path):
    RSLib = cdll.LoadLibrary(path)
    RSLib.Granule2Grid.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   c_int, \
                                   c_int, \
                                   c_float, \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   c_float, \
                                   c_float, \
                                   c_float, \
                                   c_int, \
                                   c_int, \
                                   c_float]
    
    RSLib.Granule2GridForBRDF.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   c_int, \
                                   c_int, \
                                   c_float, \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                   c_float, \
                                   c_float, \
                                   c_float, \
                                   c_int, \
                                   c_int, \
                                   c_float]
    
    RSLib.Granule2GridforVAA.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                         c_int, \
                                         c_int, \
                                         c_float, \
                                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
                                         c_float, \
                                         c_float, \
                                         c_float, \
                                         c_int, \
                                         c_int, \
                                         c_float]
    return RSLib

if __name__ == '__main__':
    RSLib = RSLibImport(r'C:\Users\1\Desktop\py\Match\RSLib_test.so')
    ChlaFile = glob.glob(Chlapath + '/*.nc')
    LON = np.zeros([nRow1,nCol1])
    LAT = np.zeros([nRow1,nCol1])
    for i in np.arange(nCol1) :
        LON[:,i] = -179.979 + 0.041666*i
    for i in np.arange(nRow1):
        LAT[i] = 89.979 - 0.041666 * i
    for k , ChlaFile in enumerate(ChlaFile):
        SDs = Dataset(ChlaFile)
        Chla = SDs['chlor_a'][:]
        Chla.shape = nRow1, nCol1
        sOutFile = ChlaFile.replace('.nc', suffix)
        nRow1,nCol1 = LON.shape
        np.where(Chla == Chla[100,100], fillvalue1, Chla)
        fp = open(sOutFile, 'wb')
        bSuccess = RSLib.Granule2GridForBRDF(np.float32(Chla), np.float32(LON), np.float32(LAT), nRow1, nCol1,
                                            fillvalue1, OUT, COUNT,COUNTFILL, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)
        fp.close()

        sHdrFile = sOutFile.replace('.dat', '.hdr')
        # sHdrFile = sHdrFile.replace('MYD35', 'MYD02')
        fp = open(sHdrFile, 'w')
        fp.write('ENVI\ndescription = {\n  File Imported into ENVI.}\n')
        fp.write('samples = {0:d}\nlines   = {1:d}\nbands   = {2:d}\n'.format(nCol2, nRow2, 1))
        fp.write(
            'header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\ndata ignore value = -1\n'
            'sensor type = Unknown\nbyte order = 0\nwavelength units = Unknown\n')

        fp.write(
            'band names = {Chla}')  
        fp.close()
        print('Finished!\n')

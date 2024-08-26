import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF

nRow1 = 3600
nCol1 = 7200
nRow2 = 721
nCol2 = 1440
lon0 = 0.0
lat0 = 90.0
dxy = 0.25
suffix = '_NDVI_0.25X0.25.dat'
fillvalue1 = -1
fillvalue2 = -1

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
    NDVIFile = r"C:\Users\1\Desktop\casestudy\data\MODIS\December\NDVI\MYD13C1.A2021361.061.2022010054537.hdf"
    LON = np.zeros([nRow1,nCol1])
    LAT = np.zeros([nRow1,nCol1])
    for i in np.arange(nCol1) :
        LON[:,i] = -179.975 + 0.05*i
    for i in np.arange(nRow1):
        LAT[i] = 89.975 - 0.05 * i

    SDs = HDF.SD(NDVIFile)
    NDVI = SDs.select('CMG 0.05 Deg 16 days NDVI').get()
    NDVI  = np.where(NDVI  != SDs.select('CMG 0.05 Deg 16 days NDVI').attributes(full=1).get('_FillValue')[0],
                               NDVI  * 0.0001,fillvalue1)

    sOutFile = NDVIFile.replace('.hdf', suffix)
    nRow1,nCol1 = LON.shape
    fp = open(sOutFile, 'wb')

    bSuccess = RSLib.Granule2GridForBRDF(np.float32(NDVI), np.float32(LON), np.float32(LAT), nRow1, nCol1,
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
        'band names = {NDVI}')  
    fp.close()
    print(NDVIFile+'Finished!\n')

import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import glob
from tqdm import tqdm

BRDFpath =r'C:\Users\1\Desktop\casestudy\data\MODIS\September\BRDF\BRDFC1origin'

nRow1 = 3600
nCol1 = 7200
nRow2 = 721
nCol2 = 1440
lon0 = 0.0
lat0 = 90.0
dxy = 0.25
suffix = '_BRDF_0.25X0.25.dat'
fillvalue1 = 0
fillvalue2 = 0

OUT = np.zeros((nRow2, nCol2), dtype=np.float32)
COUNT = np.zeros((nRow2, nCol2), dtype=np.float32)
COUNTFILL = np.zeros((nRow2, nCol2), dtype=np.float32)

def RSLibImport(path):
    RSLib = cdll.LoadLibrary(path)
    RSLib.Granule2GridForFill.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), \
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

    filelist = glob.glob(BRDFpath + '/*.hdf')


    para = list(np.arange(7))

    LON = np.zeros([nRow1,nCol1])
    LAT = np.zeros([nRow1,nCol1])
    for i in np.arange(nCol1) :
        LON[:,i] = -179.975 + 0.05*i
    for i in np.arange(nRow1):
        LAT[i] = 89.975 - 0.05 * i


    for k, sHdf in tqdm(enumerate(filelist)):

        SDs = HDF.SD(filelist[k])
        # para for band
        for i in np.arange(7):
            para[i] = list(np.arange(3))
            for j in np.arange(3):
                if j <=3:
                      para[i][j]  = SDs.select('BRDF_Albedo_Parameter{0}_Band{1}'.format(j+1,i+1 )).get()
                      a = SDs.select('BRDF_Albedo_Parameter{0}_Band{1}'.format(j+1,i+1 )).attributes(full=1).get('_FillValue')[0]
                      para[i][j]  = np.where(para[i][j]  != SDs.select('BRDF_Albedo_Parameter{0}_Band{1}'.format(j+1,i+1 )).attributes(full=1).get('_FillValue')[0],
                                             para[i][j]  * 0.001,fillvalue1)
        #para for vis and nir
        # for i in np.arange(3):
        #     para_all[i] = list(np.arange(2))
        #     for j in np.arange(2):
        #         if j == 0:
        #             para_all[i][j] = SDs.select('BRDF_Albedo_Parameter{}_vis'.format(i + 1)).get()
        #             para_all[i][j]  = np.where(para_all[i][j]  != SDs.select('BRDF_Albedo_Parameter{}_vis'.format(i+1)).attributes(full=1).get('_FillValue')[0],
        #                                    para_all[i][j]  * 0.001,fillvalue1)
        #         if j == 1:
        #             para_all[i][j] = SDs.select('BRDF_Albedo_Parameter{}_nir'.format(i + 1)).get()
        #             para_all[i][j] = np.where(para_all[i][j] != SDs.select(
        #                 'BRDF_Albedo_Parameter{}_nir'.format(i + 1)).attributes(full=1).get('_FillValue')[0],
        #                                       para_all[i][j] * 0.001, fillvalue1)

        # year = os.path.basename(sHdf)[9:13]
        # day = os.path.basename(sHdf)[13:16]
        # dir = os.listdir(sHdf)
        # sOutFile = os.path.join(dir, year,day,suffix)
        sOutFile = filelist[k].replace('.hdf', suffix)
        nRow1,nCol1 = LON.shape
        fp = open(sOutFile, 'wb')

        for i in np.arange(7):
            for j in np.arange(3):
                bSuccess = RSLib.Granule2GridForBRDF(np.float32(para[i][j]), np.float32(LON), np.float32(LAT), nRow1, nCol1,
                                          fillvalue1, OUT, COUNT,COUNTFILL,lon0, lat0, dxy, nRow2, nCol2, fillvalue2)

                fp.write(OUT)


        fp.close()

        sHdrFile = sOutFile.replace('.dat', '.hdr')
        # sHdrFile = sHdrFile.replace('MYD35', 'MYD02')
        fp = open(sHdrFile, 'w')
        fp.write('ENVI\ndescription = {\n  File Imported into ENVI.}\n')
        fp.write('samples = {0:d}\nlines   = {1:d}\nbands   = {2:d}\n'.format(nCol2, nRow2, 21))
        fp.write(
            'header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\ndata ignore value = 0\n'
            'sensor type = Unknown\nbyte order = 0\nwavelength units = Unknown\n')

        fp.write(
            'band names = {B1P1,B1P2,B1P3,B2P1,B2P2,B2P3,B3P1,B3P2,B3P3,B4P1,B4P2,B4P3,B5P1,B5P2,B5P3,B6P1,B6P2,B6P3,B7P1,B7P2,B7P3}')  # FY-3E MERSI
        # fp.write('band names = {CH24, CH25, VZA, VAA, RTime}') # FY-3D MERSI
        fp.close()
        print(sHdf+'Finished!\n')

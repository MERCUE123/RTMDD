# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.pyplot import*
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
from dateutil.parser import parse
from scipy import interpolate
import glob
from datetime import datetime, timedelta
import pandas as pd
import os
from tqdm import tqdm

#!!!
Month = 'September'

# Band1-19&26
nrow = 721
ncol = 1440
SBand = 7
MODIS = list(np.array([0.645,0.856,0.466,0.554,1.241,1.628,2.113,0.412,0.442,0.487,
                    0.530,0.547,0.666,0.678,0.747,0.867,0.904,0.936,0.935]))

MERSI = list(np.array([0.471,0.554,0.653,0.868,1.381,1.645,2.125,0.411,0.444,0.490,
                    0.556,0.670,0.709,0.746,0.865,0.905,0.936,0.940,1.030])) 

VIIRS = list(np.array([0.412,0.445,0.488,0.555,0.672,0.746,0.865,1.24,1.378,1.61,2.25])) 

MODISdic = {'Band {}'.format(i+1): value for i, value in enumerate(MODIS)}

SampleBand = ['Band 3','Band 4','Band 1','Band 2','Band 5','Band 6','Band 7']
MODIS1 = np.array([MODISdic[key] for key in SampleBand])

BRDFpath = r'C:/Users/1/Desktop/casestudy/data/MODIS/'+Month+'/BRDF/BRDFC1_Sampleto_0.25x0.25MODIS'
BRDFfiles = glob.glob(BRDFpath + '/*.dat')

## !!
Term = MERSI


if Term == MODIS:
    nBand = 19
    sOutPath = r'C:/Users/1/Desktop/casestudy/data/MODIS/'+Month+'/BRDF/BRDFC1_Sampleto_19CH_MODIS'
elif Term == MERSI:
    nBand = 19
    sOutPath = r'C:/Users/1/Desktop/casestudy/data/MODIS/'+Month+'/BRDF/BRDFC1_Sampleto_19CH_MERSI'
elif Term == VIIRS:
    nBand = 11
    sOutPath = r'C:/Users/1/Desktop/casestudy/data/MODIS/'+Month+'/BRDF/BRDFC1_Sampleto_11CH_VIIRS'


# Term = MERSI
# sOutPath = r'C:/Users/1/Desktop/casestudy/data/MODIS\BRDF\BRDF_Sampleto_19CH_FY3D'

Sp1 =list(np.arange(7))
Sp2 =list(np.arange(7))
Sp3 =list(np.arange(7))

p1 = np.zeros([nBand,nrow,ncol],dtype=np.float32)
p2 = np.zeros([nBand,nrow,ncol],dtype=np.float32)
p3 = np.zeros([nBand,nrow,ncol],dtype=np.float32)
for k,BRDFfile in tqdm(enumerate(BRDFfiles)):
    print(BRDFfile)
    BRDF = np.fromfile(BRDFfile, dtype=np.float32)
    BRDF.shape = 21, 721, 1440
    
    sOutFile = os.path.basename(BRDFfiles[k].replace('.HDF', '.dat'))
    sOutFile = os.path.join(sOutPath ,sOutFile)
    fp = open(sOutFile, 'wb')
    j=0
    # for i in np.arange(7):
    #     j=i*3
    #     Sp1[i] = BRDF[j,:,:]
    #     Sp2[i] = BRDF[j+1, :, :]
    #     Sp3[i] = BRDF[j+2, :, :]
    for i in np.arange(7):
        if i ==0:
            Sp1[i] = BRDF[6,:,:]
            Sp2[i] = BRDF[7, :, :]
            Sp3[i] = BRDF[8, :, :]
        if i ==1 :
            Sp1[i] = BRDF[9,:,:]
            Sp2[i] = BRDF[10, :, :]
            Sp3[i] = BRDF[11, :, :]
        if i ==2:
            Sp1[i] = BRDF[0,:,:]
            Sp2[i] = BRDF[1, :, :]
            Sp3[i] = BRDF[2, :, :]
        if i == 3:
            Sp1[i] = BRDF[3, :, :]
            Sp2[i] = BRDF[4, :, :]
            Sp3[i] = BRDF[5, :, :]
        if i >=4:
            j = i*3
            Sp1[i] = BRDF[j,:,:]
            Sp2[i] = BRDF[j+1, :, :]
            Sp3[i] = BRDF[j+2, :, :]

    for i in tqdm(np.arange(nrow)):
        # print('epoch '+str(i))
        ## debug专用 运行注释掉
        # if i < 222 : continue

        for j in np.arange(ncol):
            ## debug专用 运行注释掉
            # if j < 364 :continue
            x = MODIS1
            y1 = []
            y2 = []
            y3 = []
            for p in np.arange(SBand):

                y1.append(Sp1[p][i, j])

                y2.append(Sp2[p][ i, j])

                y3.append(Sp3[p] [i, j])
            ## 画图专用 运行注释掉
            # a = sum(y1)
            # if sum(y1 ) <=0.00001 :
            #     continue
            # fp1 = interpolate.interp1d(x, y1, kind='quadratic',fill_value="extrapolate")
            # fp2 = interpolate.interp1d(x, y2, kind='quadratic',fill_value="extrapolate")
            # fp3 = interpolate.interp1d(x, y3, kind='quadratic',fill_value="extrapolate")
            # fp1 = interpolate.UnivariateSpline(x, y1,s=0,k=2)
            # fp2 = interpolate.UnivariateSpline(x, y2,s=0,k=2)
            # fp3 = interpolate.UnivariateSpline(x, y3,s=0,k=2)
            fp1 =  np.poly1d(np.polyfit(x,y1,2))
            fp2 = np.poly1d(np.polyfit(x,y2,2))
            fp3 = np.poly1d(np.polyfit(x,y3,2))
            Ep1 = fp1(np.array(Term))
            Ep2 = fp2(np.array(Term))
            Ep3 = fp3(np.array(Term))
            # q = BRDF[0,i,j]
            # p = Sp1[0][i,j]
            for p in np.arange(nBand):
                p1[p, i, j] = Ep1[p]
                p2[p, i, j] = Ep2[p]
                p3[p, i, j] = Ep3[p]

    fp.write(p1)
    fp.write(p2)
    fp.write(p3)

    sHdrFile = sOutFile.replace('.dat', '.hdr')
    if Term == VIIRS :
        fp = open(sHdrFile, 'w')
        fp.write('ENVI\ndescription = {\n  File Imported into ENVI.}\n')
        fp.write('samples = {0:d}\nlines   = {1:d}\nbands   = {2:d}\n'.format(ncol, nrow, 33))
        fp.write(
                'header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\ndata ignore value = 0\n'
                'sensor type = Unknown\nbyte order = 0\nwavelength units = Unknown\n')

        fp.write(
                'band names = {P1B1,P1B2,P1B3,P1B4,P1B5,P1B6,B1P7,P1B8,P1B9,P1B10,P1B11,'
                'P2B1,P2B2,P2B3,P2B4,P2B5,P2B6,B1P7,P2B8,P2B9,P2B10,P2B11,'
                'P3B1,P3B2,P3B3,P3B4,P3B5,P3B6,B1P7,P3B8,P3B9,P3B10,P3B11}')  # FY-3E MERSI
    else:

        fp = open(sHdrFile, 'w')
        fp.write('ENVI\ndescription = {\n  File Imported into ENVI.}\n')
        fp.write('samples = {0:d}\nlines   = {1:d}\nbands   = {2:d}\n'.format(ncol, nrow, 57))
        fp.write(
                'header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\ndata ignore value = 0\n'
                'sensor type = Unknown\nbyte order = 0\nwavelength units = Unknown\n')

        fp.write(
                'band names = {P1B1,P1B2,P1B3,P1B4,P1B5,P1B6,B1P7,P1B8,P1B9,P1B10,P1B11,P1B12,P1B13,P1B14,P1B15,P1B16,P1B17,P1B18,P1B19,'
                'P2B1,P2B2,P2B3,P2B4,P2B5,P2B6,B1P7,P2B8,P2B9,P2B10,P2B11,P2B12,P2B13,P2B14,P2B15,P2B16,P2B17,P2B18,P2B19,'
                'P3B1,P3B2,P3B3,P3B4,P3B5,P3B6,B1P7,P3B8,P3B9,P3B10,P3B11,P3B12,P3B13,P3B14,P3B15,P3B16,P3B17,P3B18,P3B19}')  # FY-3E MERSI

        fp.close()

# -*- coding: utf-8 -*-

import sys
import numpy as np
import math
from datetime import datetime, timedelta
import scipy.constants as const
from netCDF4 import Dataset
import os, fnmatch
import glob
from tqdm import tqdm, trange
# list1 = r'C:\Users\1\Desktop\py\Match\list3D_Sample_Data.dat'
Month = 'September'
list1path = r'C:/Users/1/Desktop/casestudy/data/MERSI/'+Month+'/SampleData'
# list2path = r'C:\Users\1\Desktop\casestudy\data\/VIIRS/'+Month+'/SampleData'
list2path = r'C:/Users/1/Desktop/casestudy/data/VIIRS/'+Month+'/SampleData'
MERSI_files = glob.glob(list1path+'/*.dat')
# MERSI_files = glob.glob(list1path+'/*.dat')
VIIRS_files = glob.glob(list2path+'/*.dat')
# VIIRS_files = glob.glob(list2path+'/*.dat')

sYear = '2021'
if Month == 'March':
    sMonth = '03'
elif Month == 'June':
    sMonth = '06'
elif Month == 'September':
    sMonth = '09'
else:
    sMonth='12'

LandOutFile = r'C:/Users/1/Desktop/casestudy\data/MatchRes/'+Month+'/LandMatchTable_MERSI_VIIRS_1'+Month+'.csv'
# 0 for sea and 1 for land
landmaskfile =r'C:\Users\1\Desktop\casestudy\data\LandMask_0.25X0.25.dat'

# ERA5_file = r'C:\Users\1\Desktop\casestudy\data\ERA5\Atoms\\'
if Month == 'March':
        NDVIpath1 = r"C:\Users\1\Desktop\casestudy\data\MODIS\March\NDVI\MYD13C1.A2021057.061.2021076030617_NDVI_0.25X0.25.dat"
        NDVIpath2 = r'C:\Users\1\Desktop\casestudy\data\MODIS\March\NDVI\MYD13C1.A2021073.061.2021090083039_NDVI_0.25X0.25.dat'
        NDVIpath3 = r"C:\Users\1\Desktop\casestudy\data\MODIS\March\NDVI\MYD13C1.A2021089.061.2021107135045_NDVI_0.25X0.25.dat"

        NDVI1 = np.fromfile(NDVIpath1 , dtype=np.float32)
        NDVI2 = np.fromfile(NDVIpath2, dtype=np.float32)
        NDVI3 = np.fromfile(NDVIpath3, dtype=np.float32)
        NDVI1.shape = 721, 1440
        NDVI2.shape = 721, 1440
        NDVI3.shape = 721, 1440

if Month == 'June':
        NDVIpath1 = r"C:\Users\1\Desktop\casestudy\data\MODIS\June\NDVI\MYD13C1.A2021153.061.2021170081513_NDVI_0.25X0.25.dat"
        NDVIpath2 = r"C:\Users\1\Desktop\casestudy\data\MODIS\June\NDVI\MYD13C1.A2021169.061.2021188163527_NDVI_0.25X0.25.dat"

        NDVI1 = np.fromfile(NDVIpath1 , dtype=np.float32)
        NDVI2 = np.fromfile(NDVIpath2, dtype=np.float32)
        NDVI1.shape = 721, 1440
        NDVI2.shape = 721, 1440
        
if Month == 'September':
        NDVIpath1 = r"C:\Users\1\Desktop\casestudy\data\MODIS\September\NDVI\MYD13C1.A2021249.061.2021266063437_NDVI_0.25X0.25.dat"
        NDVIpath2 = r"C:\Users\1\Desktop\casestudy\data\MODIS\September\NDVI\MYD13C1.A2021265.061.2021282064443_NDVI_0.25X0.25.dat"
        NDVI2 = np.fromfile(NDVIpath2 , dtype=np.float32)
        NDVI1 = np.fromfile(NDVIpath1 , dtype=np.float32)
        NDVI1.shape = 721, 1440
        NDVI2.shape = 721, 1440
if Month == 'December':
        NDVIpath1 = r"C:\Users\1\Desktop\casestudy\data\MODIS\December\NDVI\MYD13C1.A2021329.061.2021346065844_NDVI_0.25X0.25.dat"
        NDVIpath2 = r"C:\Users\1\Desktop\casestudy\data\MODIS\December\NDVI\MYD13C1.A2021345.061.2022004004026_NDVI_0.25X0.25.dat"
        NDVIpath3 = r"C:\Users\1\Desktop\casestudy\data\MODIS\December\NDVI\MYD13C1.A2021361.061.2022010054537_NDVI_0.25X0.25.dat"

        NDVI1 = np.fromfile(NDVIpath1 , dtype=np.float32)
        NDVI2 = np.fromfile(NDVIpath2, dtype=np.float32)
        NDVI3 = np.fromfile(NDVIpath3, dtype=np.float32)
        NDVI1.shape = 721, 1440
        NDVI2.shape = 721, 1440
        NDVI3.shape = 721, 1440
sPrefix = '' # 'S': Sea; 'L': Land

nRow = 721
nCol = 1440
nRow1 = 120 # -60°S~60°N
nRow2 = 601

nBand_MERSI = 25 # 19 bands + VZA + VAA + SZA + SAA + Rtime



nVZAIndex1 = 19
nVAAIndex1 = 20
nTimeIndex1 = 21
nCLMIndex1 = 22
nSZAIndex1 = 23
nSAAIndex1 = 24


nBand_VIIRS = 19  # 13 bands + VZA + VAA + SZA + SAA + Rtime

nVZAIndex2 = 13
nVAAIndex2 = 14
nTimeIndex2 = 15
nCLMIndex2 = 16
nSZAIndex2 = 17
nSAAIndex2 = 18


time0 = datetime(2021, 1, 1, 0, 0)
dtl = 15       # delta time limit in minutes
dtl /= 1440 # in days
ERA5TimeList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

Landmask = np.fromfile(landmaskfile, dtype=np.byte)
Landmask.shape = nRow, nCol

# NDVI = np.fromfile(NDVI_file, dtype=np.float32)[0:nRow*nCol]
# NDVI.shape = nRow, nCol

# MERSI_files = fnmatch.filter(os.listdir(list1path), '*.dat')

# Obtain the time range for each VIIRS file
TMin = []
TMax = []


for sVIIRSFile in tqdm(VIIRS_files, desc='Reading VIIRS files'):
    VIIRS = np.fromfile(sVIIRSFile, dtype=np.float32)
    VIIRS.shape = nBand_VIIRS, nRow, nCol
    VIIRS0 = VIIRS[:,nRow1:nRow2,:]
    if (VIIRS0[nTimeIndex2,:,:]).max()<0.00001:
        TMin.append(999.0) 
        TMax.append(0.0)
    else:
        TMin.append(VIIRS0[nTimeIndex2,VIIRS0[nTimeIndex2,:,:]>0].min() - dtl/2) # minus/plus half of the matching window time
        TMax.append(VIIRS0[nTimeIndex2,VIIRS0[nTimeIndex2,:,:]>0].max() + dtl/2)

count = 0
fp = open(LandOutFile, 'w')


fp.write('Land/Sea,ERA5Year,ERA5Month,ERA5Day,ERA5Hour,'
         'MERSI_Day,MERSI_Hour,MERSI_Minute,VIIRSDay,VIIRSHour,VIIRSMinute,'
         'iPos,jPos,DeltaTime1,DeltaTime2,'
         'B1_MERSI,B2_MERSI,B3_MERSI,B4_MERSI,'
         'B5_MERSI,B6_MERSI,B7_MERSI,B8_MERSI,'
         'B9_MERSI,B10_MERSI,B11_MERSI,B12_MERSI,'
         'B13_MERSI,B14_MERSI,B15_MERSI,B16_MERSI,'
         'B17_MERSI,B18_MERSI,B19_MERSI,'
         'B1_VIIRS,B2_VIIRS,B3_VIIRS,B4_VIIRS,'
         'B5_VIIRS,B6_VIIRS,B7_VIIRS,B8_VIIRS,'
         'B9_VIIRS,B10_VIIRS,B11_VIIRS,'
         'VZA_MERSI,VAA_MERSI,SZA_MERSI,SAA_MERSI,CLM_MERSI,'
         'VZA_VIIRS,VAA_VIIRS,SZA_VIIRS,SAA_VIIRS,CLM_VIIRS,NDVI,dTime\n')


year0 = -1.0
month0 = -1.0
day0 = -1.0
hour0 = -1.0
for sMERSIFile in tqdm(MERSI_files, desc='Reading MERSI files'):
    print(sMERSIFile, end='...\n')
    MERSI = np.fromfile(sMERSIFile, dtype=np.float32)
    MERSI.shape = nBand_MERSI, nRow, nCol
    MERSI0 = MERSI[:,nRow1:nRow2,:]
    if (MERSI0[nTimeIndex1,:,:]).max()<=0: continue
    t1 = MERSI0[nTimeIndex1,MERSI0[nTimeIndex1,:,:]>0].min() - dtl/2
    t2 = MERSI0[nTimeIndex1,MERSI0[nTimeIndex1,:,:]>0].max() + dtl/2

    for k, sVIIRSFile in tqdm(enumerate(VIIRS_files),desc='Matching MERSI and VIIRS'):
        if TMin[k]>t2 or t1>TMax[k]: continue # no time overlap
        print(sVIIRSFile, end='...\n')
        VIIRS = np.fromfile(sVIIRSFile, dtype=np.float32)
        VIIRS.shape = nBand_VIIRS, nRow, nCol
        for i in range (nRow1,nRow2):
            for j in range(1, nCol-1):
                
                # 点在3x3的区域内都有值
                if sum(sum(MERSI[0,i-1:i+2,j-1:j+2]<=0))>0 or sum(sum(VIIRS[0,i-1:i+2,j-1:j+2]<=0))>0: continue
                
                # 时间匹配
                dTime=  math.fabs(MERSI[nTimeIndex1, i, j] - VIIRS[nTimeIndex2, i, j])
                dTimeseconds = dTime*1440*60
                if dTime > dtl: continue
                # 云的判定
                if sum(sum(MERSI[nCLMIndex1][i-1:i+2,j-1:j+2] <=3.8)) > 0 :continue
                if sum(sum(VIIRS[nCLMIndex2][i-1:i+2,j-1:j+2] <=3.8)) > 0 :continue

                if sum(sum(Landmask[i-1:i+2,j-1:j+2] < 0.5))==9:
                # sPrefix = 'S'    #海洋
                    continue
                elif sum(sum(Landmask[i-1:i+2,j-1:j+2] > 0.5))==9:
                    sPrefix = 'L'    #陆地
                else: continue
                # 角度判定
                # SZA角度限制
                if MERSI[nSZAIndex1, i, j] >65 : continue
                if VIIRS[nSZAIndex2, i, j] >65 : continue
                # VZA角度限制
                if MERSI[nVZAIndex1, i, j] >65 : continue
                if VIIRS[nVZAIndex2, i, j] >65 : continue
                c1 = math.cos(MERSI[nVZAIndex1, i, j]*const.pi/180)
                c2 = math.cos(VIIRS[nVZAIndex2,i,j]*const.pi/180)
                if math.fabs(c1/c2-1) > 0.1 : continue
                # VAA角度限制
                if VIIRS[nVAAIndex2,i,j] <0 :
                    VIIRS[nVAAIndex2,i,j] +=360
                if math.fabs(MERSI[nVAAIndex1, i, j] - VIIRS[nVAAIndex2,i,j]) > 10 : continue


                # -1 for tmep use
                time1 = time0 + timedelta(days=float(MERSI[nTimeIndex1, i, j]) + (ERA5TimeList[1]-ERA5TimeList[0])/(2*24)) # ?
                # hour1 = time1.hour + time1.minute/60 + time1.second/3600
                time2 = time0 + timedelta(days=float(VIIRS[nTimeIndex2, i, j]) + (ERA5TimeList[1]-ERA5TimeList[0])/(2*24)) # ？
                if time1.hour!=time2.hour: continue # the same ERA5 time
                
                timeFY_3D = time0+timedelta(days=float(MERSI[nTimeIndex1,i,j]))
                timeVIIRS = time0+timedelta(days=float(VIIRS[nTimeIndex2,i,j]))


                ERA5Time = datetime(int(time1.year), int(time1.month), int(time1.day), int(time1.hour), 0)
                # if ERA5Time.day<10: sDay = '0' + str(ERA5Time.day)
                # else: sDay = str(ERA5Time.day)
                sDay = '%02d'%ERA5Time.day
                # if ERA5Time.hour<10: sHour = '0' + str(ERA5Time.hour)
                # else: sHour = str(ERA5Time.hour)
                sHour = '%02d'%ERA5Time.hour
                
                lon = 0.25 * j
                lat = 90 - 0.25 * i

                dt1 = time0 + timedelta(days=float(MERSI[nTimeIndex1,i,j])) - ERA5Time
                dt2 = time0 + timedelta(days=float(VIIRS[nTimeIndex2, i, j])) - ERA5Time
    


                fp.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},'.format(sPrefix, sYear, sMonth, sDay, '%d'%ERA5Time.hour, timeFY_3D.day, '%d'%timeFY_3D.hour, '%d'%timeFY_3D.minute, timeVIIRS.day, '%d'%timeVIIRS.hour, '%d'%timeVIIRS.minute, '%d'%i, '%d'%j, '%.1f'%(1440*dt1.days+dt1.seconds/60+dt1.microseconds/60000000), '%.1f'%(1440*dt2.days+dt2.seconds/60+dt2.microseconds/60000000)))

                # March
                if Month == 'March':
                    if  ERA5Time.day <= 7 :
                        NDVI = NDVI1
                    elif  21>= ERA5Time.day > 7 :
                        NDVI = NDVI2
                    else :
                        NDVI = NDVI3
                # June
                if Month == 'June':
                    if  ERA5Time.day <= 11 :
                        NDVI = NDVI1
                    elif ERA5Time.day == 31:
                        NDVI = NDVI1
                    else :
                        NDVI = NDVI2
                        
                # September
                if Month == 'September':
                    if  ERA5Time.day <= 13 :
                        NDVI = NDVI1
                    else:
                        NDVI = NDVI2
                
                # December
                if Month == 'December':
                    if  ERA5Time.day.day <= 3 :
                        NDVI = NDVI1
                    elif  19>= ERA5Time.day > 3 :
                        NDVI = NDVI2
                    else :
                        NDVI = NDVI3

                for Band in range(19):
                    # MERSI定标完差0.01倍
                    MERSI[Band, i, j] = float(MERSI[Band, i, j] )*0.01
                    fp.write('{0},'.format('%.7f' % MERSI[Band, i, j]))
                for Band in range(11):
                    fp.write('{0},'.format('%.7f' % VIIRS[Band, i, j]))

                fp.write('{0},'.format('%.1f'%MERSI[nVZAIndex1,i,j])) 
                fp.write('{0},'.format('%.1f'%MERSI[nVAAIndex1,i,j])) 
                fp.write('{0},'.format('%.1f'%MERSI[nSZAIndex1,i,j]))
                fp.write('{0},'.format('%.1f'%MERSI[nSAAIndex1,i,j]))
                fp.write('{0},'.format('%.2f'%MERSI[nCLMIndex1,i,j]))

                fp.write('{0},'.format('%.1f'%VIIRS[nVZAIndex2,i,j])) 
                fp.write('{0},'.format('%.1f'%VIIRS[nVAAIndex2,i,j])) 
                fp.write('{0},'.format('%.1f'%VIIRS[nSZAIndex2,i,j])) 
                fp.write('{0},'.format('%.1f'%VIIRS[nSAAIndex2,i,j]))
                fp.write('{0},'.format('%.2f' % VIIRS[nCLMIndex2,i,j]))
                fp.write('{0},'.format('%.2f' % NDVI[i,j]))
                fp.write('{0},'.format('%.1f'%dTimeseconds))
                fp.write('\n')
                count += 1
                print('Done.')
    print('OK.')
    print('count: ' + str(count))
fp.close()
print('Total count: ' + str(count))

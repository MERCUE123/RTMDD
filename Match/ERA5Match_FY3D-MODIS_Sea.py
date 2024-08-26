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
Month = 'March'
list1path = r'C:\Users\1\Desktop\casestudy\data\/MERSI/'+Month+'/SampleData'
list2path = r'C:\Users\1\Desktop\casestudy\data\/MODIS/'+Month+'/SampleData'

sYear = '2021'
if Month == 'March':
    sMonth = '03'
elif Month == 'June':
    sMonth = '06'

OutFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/LandMatchTable_MERSI_MODIS_03_Sea.csv'
# 0 for sea and 1 for land
landmaskfile =r'C:\Users\1\Desktop\casestudy\data\LandMask_0.25X0.25.dat'

# ERA5_file = r'C:\Users\1\Desktop\casestudy\data\ERA5\Atoms\\'

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


nBand_MODIS = 27  # 22 bands + VZA + VAA + SZA + SAA + Rtime

nVZAIndex2 = 21
nVAAIndex2 = 22
nTimeIndex2 = 23
nCLMIndex2 = 24
nSZAIndex2 = 25
nSAAIndex2 = 26


time0 = datetime(2021, 1, 1, 0, 0)
dtl = 5       # delta time limit in minutes
dtl /= 1440 # in days
ERA5TimeList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

Landmask = np.fromfile(landmaskfile, dtype=np.byte)
Landmask.shape = nRow, nCol



# MERSI_files = fnmatch.filter(os.listdir(list1path), '*.dat')
MERSI_files = glob.glob(list1path+'/*.dat')
MODIS_files = glob.glob(list2path+'/*.dat')

# Obtain the time range for each MODIS file
TMin = []
TMax = []


for sMODISFile in tqdm(MODIS_files, desc='Reading MODIS files'):
    MODIS = np.fromfile(sMODISFile, dtype=np.float32)
    MODIS.shape = nBand_MODIS, nRow, nCol
    MODIS0 = MODIS[:,nRow1:nRow2,:]
    if (MODIS0[nTimeIndex2,:,:]).max()<0.00001:
        TMin.append(999.0) 
        TMax.append(0.0)
    else:
        TMin.append(MODIS0[nTimeIndex2,MODIS0[nTimeIndex2,:,:]>0].min() - dtl/2) # minus/plus half of the matching window time
        TMax.append(MODIS0[nTimeIndex2,MODIS0[nTimeIndex2,:,:]>0].max() + dtl/2)

count = 0
fp = open(OutFile, 'w')


fp.write('Land/Sea,ERA5Year,ERA5Month,ERA5Day,ERA5Hour,'
         'MERSI_Day,MERSI_Hour,MERSI_Minute,MODISDay,MODISHour,MODISMinute,'
         'iPos,jPos,DeltaTime1,DeltaTime2,'
         'B1_MERSI,B1_MODIS,B2_MERSI,B2_MODIS,B3_MERSI,B3_MODIS,B4_MERSI,B4_MODIS,'
         'B5_MERSI,B5_MODIS,B6_MERSI,B6_MODIS,B7_MERSI,B7_MODIS,B8_MERSI,B8_MODIS,'
         'B9_MERSI,B9_MODIS,B10_MERSI,B10_MODIS,B11_MERSI,B11_MODIS,B12_MERSI,B12_MODIS,'
         'B13_MERSI,B13_MODIS,B14_MERSI,B14_MODIS,B15_MERSI,B15_MODIS,B16_MERSI,B16_MODIS,'
         'B17_MERSI,B17_MODIS,B18_MERSI,B18_MODIS,B19_MERSI,B19_MODIS,'
         'VZA_MERSI,VAA_MERSI,SZA_MERSI,SAA_MERSI,CLM_MERSI,'
         'VZA_MODIS,VAA_MODIS,SZA_MODIS,SAA_MODIS,CLM_MODIS\n')


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

    for k, sMODISFile in tqdm(enumerate(MODIS_files),desc='Matching MERSI and MODIS'):
        if TMin[k]>t2 or t1>TMax[k]: continue # no time overlap
        print(sMODISFile, end='...\n')
        MODIS = np.fromfile(sMODISFile, dtype=np.float32)
        MODIS.shape = nBand_MODIS, nRow, nCol
        for i in range (nRow1,nRow2):
            for j in range(1, nCol-1):
                # 点在3x3的区域内都有值
                if sum(sum(MERSI[0,i-1:i+2,j-1:j+2]<=0))>0 or sum(sum(MODIS[0,i-1:i+2,j-1:j+2]<=0))>0: continue
                 # 云的判定
                
                if sum(sum(MERSI[nCLMIndex1][i-1:i+2,j-1:j+2] <=3.8)) > 0 :continue
                if sum(sum(MODIS[nCLMIndex2][i-1:i+2,j-1:j+2] <=3.8)) > 0 :continue

                if sum(sum(Landmask[i-1:i+2,j-1:j+2] < 0.5))==9:
                    sPrefix = 'S'    #海洋
                elif sum(sum(Landmask[i-1:i+2,j-1:j+2] > 0.5))==9:
                    # sPrefix = 'L'    #陆地
                    continue
                else: continue
                # 角度判定
                # SZA角度限制
                if MERSI[nSZAIndex1, i, j] >65 : continue
                if MODIS[nSZAIndex2, i, j] >65 : continue
                # VZA角度限制
                if MERSI[nVZAIndex1, i, j] >65 : continue
                if MODIS[nVZAIndex2, i, j] >65 : continue
                c1 = math.cos(MERSI[nVZAIndex1, i, j]*const.pi/180)
                c2 = math.cos(MODIS[nVZAIndex2,i,j]*const.pi/180)
                if math.fabs(c1/c2-1) > 0.1 : continue
                # VAA角度限制
                if MODIS[nVAAIndex2,i,j] <0 :
                    MODIS[nVAAIndex2,i,j] +=360
                if math.fabs(MERSI[nVAAIndex1, i, j] - MODIS[nVAAIndex2,i,j]) > 10 : continue
                # 时空匹配
                if math.fabs(MERSI[nTimeIndex1, i, j] - MODIS[nTimeIndex2, i, j]) > dtl: continue
                
              


                
                
              



                # -1 for tmep use
                time1 = time0 + timedelta(days=float(MERSI[nTimeIndex1, i, j]) + (ERA5TimeList[1]-ERA5TimeList[0])/(2*24)) # ?
                # hour1 = time1.hour + time1.minute/60 + time1.second/3600
                time2 = time0 + timedelta(days=float(MODIS[nTimeIndex2, i, j]) + (ERA5TimeList[1]-ERA5TimeList[0])/(2*24)) # ？
                if time1.hour!=time2.hour: continue # the same ERA5 time
                
                timeFY_3D = time0+timedelta(days=float(MERSI[nTimeIndex1,i,j]))
                timeMODIS = time0+timedelta(days=float(MODIS[nTimeIndex2,i,j]))


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
                dt2 = time0 + timedelta(days=float(MODIS[nTimeIndex2, i, j])) - ERA5Time
    


                fp.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},'.format(sPrefix, sYear, sMonth, sDay, '%d'%ERA5Time.hour, timeFY_3D.day, '%d'%timeFY_3D.hour, '%d'%timeFY_3D.minute, timeMODIS.day, '%d'%timeMODIS.hour, '%d'%timeMODIS.minute, '%d'%i, '%d'%j, '%.1f'%(1440*dt1.days+dt1.seconds/60+dt1.microseconds/60000000), '%.1f'%(1440*dt2.days+dt2.seconds/60+dt2.microseconds/60000000)))


                for Band in range(19):
                    # MERSI定标完差0.01倍
                    MERSI[Band, i, j] = float(MERSI[Band, i, j] )*0.01
                    fp.write('{0},'.format('%.7f' % MERSI[Band, i, j]))
                    # 剔除13H 和14H 通道
                    if Band <=12:
                        fp.write('{0},'.format('%.7f' % MODIS[Band, i, j]))
                    if Band == 13:
                        fp.write('{0},'.format('%.7f' % MODIS[Band+1, i, j]))
                    if Band >= 14 :
                        fp.write('{0},'.format('%.7f' % MODIS[Band+2, i, j]))

                fp.write('{0},'.format('%.1f'%MERSI[nVZAIndex1,i,j])) 
                fp.write('{0},'.format('%.1f'%MERSI[nVAAIndex1,i,j])) 
                fp.write('{0},'.format('%.1f'%MERSI[nSZAIndex1,i,j]))
                fp.write('{0},'.format('%.1f'%MERSI[nSAAIndex1,i,j]))
                fp.write('{0},'.format('%.2f'%MERSI[nCLMIndex1,i,j]))

                fp.write('{0},'.format('%.1f'%MODIS[nVZAIndex2,i,j])) 
                fp.write('{0},'.format('%.1f'%MODIS[nVAAIndex2,i,j])) 
                fp.write('{0},'.format('%.1f'%MODIS[nSZAIndex2,i,j])) 
                fp.write('{0},'.format('%.1f'%MODIS[nSAAIndex2,i,j]))
                fp.write('{0},'.format('%.2f' % MODIS[nCLMIndex2,i,j]))
                fp.write('\n')
                count += 1
                print('Done.')
    print('OK.')
fp.close()
print('Total count: ' + str(count))

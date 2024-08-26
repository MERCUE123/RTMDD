import sys
import os
import numpy as np
from ctypes import *
import pyhdf.SD as HDF
import pandas as pd
from datetime import datetime, timedelta
import h5py as h5py
from dateutil.parser import parse
import matplotlib.pyplot as plt
import glob
#数据列表路径并读取
Month = 'March'
# sListFileDay_Night = 'D:/MERSI/'+Month+'/GEO1K_DayAll.dat'

sListFile = r'E:/MERSI/'+Month+'/GEO1K_Day.dat'
GEOpath = r'E:/MERSI/'+Month+'/1KM&GEO&CLM'
list1 = open(sListFile, 'w')
geofile = glob.glob(GEOpath +'/*GEO1K*.HDF')
# geofile = glob.glob(GEOpath +'/*2021090[123456]*GEO1K*.HDF')
list1.write('\n'.join(geofile))
list1.close()
sListFile = r'E:/MERSI/'+Month+'/GEO1K_Day.dat'
sOutPath = r'C:/Users/1/Desktop/casestudy/data/MERSI/'+Month+'/SampleData'
# 初始化信息
nRow2 = 721  #500
nCol2 = 1440 #600
dxy = 0.25
lon0 = 0.0
lat0 = 90.0
suffix = '_0.25X0.25.dat'

fillvalue1 = 0
fillvalue2 = 0


time0 = datetime(2021, 1, 1, 0, 0)
OUT = np.zeros((nRow2, nCol2), dtype=np.float32)
COUNT = np.zeros((nRow2, nCol2), dtype=np.float32)
COUNTFILL = np.zeros((nRow2, nCol2), dtype=np.float32)

## Day Night Divide


# fp = open(sListFileDay_Night, 'r')
# filelist = [x.rstrip() for x in fp]
# fp.close()
# sOutFile = sListFileDay_Night.replace('DayNight', 'Day')
# fp = open(sOutFile, 'w')
# for k, sHdf in enumerate(filelist):
#     if 'tmp' in filelist[k]:
#         continue
#     HDF = h5py.File(filelist[k], 'r')
#     # D:\FY3D\1KM&GEO\FY3D_MERSI_GBAL_L1_20210302_2355_GEO1K_MS.HDF
#     flag = HDF['Timedata/DayNightFlag'][:]
#     # 0 代表白天
#     if np.sum(flag) == 0:
#         fp.write(str(sHdf))
#         fp.write('\n')

# fp.close()


fp = open(sListFile, 'r')
filelist = [x.rstrip() for x in fp]
fp.close()

def RSLibImport(path):
    RSLib = cdll.LoadLibrary(path)
    RSLib.Granule2Grid.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                                   c_int,\
                                   c_int,\
                                   c_float,\
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),\
                                   c_float,\
                                   c_float,\
                                   c_float,\
                                   c_int,\
                                   c_int,\
                                   c_float]
    
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

def find1KM(keyword, path):
    file = []
    files = os.listdir(path)
    for i in files:
        i = os.path.join(path,i) #合并路径与文件名
        # if os.path.isfile(i):#判断是否为文件
            # print(i)
        if keyword in os.path.basename(i):
                print('Find Match File '+keyword)  # 绝对路径
                file = i
                break
    if file == []:
        print('Find No Match File with '+keyword)
    return file

def read_Radiance_RTime(FY3Dpath):


    # 读取HDF
    if os.path.basename(FY3Dpath) in os.listdir(os.path.dirname(FY3Dpath)):

        HDF = h5py.File(FY3Dpath, 'r')
        E0 = h5py.AttributeManager(HDF).get('Solar_Irradiance')
        E0 = E0 / 3.1415
        RefSB1KM = np.array(HDF['Data/EV_1KM_RefSB'][:],dtype=np.float32)
        RefSB250 = np.array(HDF['Data/EV_250_Aggr.1KM_RefSB'][:],dtype=np.float32)
        N, row, col = RefSB250.shape
        cali = HDF['Calibration/VIS_Cal_Coeff'][:]  # 定标系数
        # BandRadiance2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        BandRadiance = list(np.arange(0, 19))
        # 存放通道数据的矩阵,共19层
        for i in range(len(BandRadiance)):
            if i <= 3:
                RefSB250t = np.where((0<RefSB250[i, :, :])&(RefSB250[i, :, :]<4095),
                                           (HDF['Calibration/VIS_Cal_Coeff'][i,1]*RefSB250[i,:,:]+
                                            HDF['Calibration/VIS_Cal_Coeff'][i,0])*E0[i],0) #cali2在可见光为0，故舍去
                BandRadiance[i] = RefSB250t

            if 4 <= i <= 18:
                temp = HDF['Calibration/VIS_Cal_Coeff'][i, 0]
                RefSB1KMt = np.where((0<RefSB1KM[i-4, :, :])&(RefSB1KM[i-4, :, :]<4095),
                                               (HDF['Calibration/VIS_Cal_Coeff'][i, 1] * RefSB1KM[i-4, :, :] +
                                             HDF['Calibration/VIS_Cal_Coeff'][i, 0]) * E0[i], 0)  # cali2在可见光为0，故舍去
                BandRadiance[i] = RefSB1KMt

        # RTime
        RTime = np.zeros((row, col), dtype=np.float32)
        time1 = parse(HDF.attrs['Observing Beginning Date'] + b' ' + HDF.attrs['Observing Beginning Time'])
        time2 = parse(HDF.attrs['Observing Ending Date'] + b' ' + HDF.attrs['Observing Ending Time'])

        linescantime = (time2 - time1) / row
        # for i in range(0, row, 10):
        #     dt = time1 + timedelta(days=float(EVscantime[i // 10] / 86400)) - time0
        #     RTime[i:i + 10, :] = np.float32(dt.days + (dt.seconds + dt.microseconds / 1000000) / 86400.0)  # in days
        for i in range(row):
            dtime = time1 + i * linescantime - time0
            RTime[i, :] = np.float32(dtime.days + dtime.seconds / 86400.0)  # in days
            # timetest = time0 + timedelta(days=float(RTime[0, 0]))
        # timetest = time0 + timedelta(days=float(RTime[500, 500]))
        HDF.close()

        return BandRadiance, RTime
    else:
        row = 2000
        col = 2048
        BandRadiance = np.zeros((19,row,col),dtype=np.float32)
        RTime = np.zeros((row, col), dtype=np.float32)
        print('No Match File')
        return BandRadiance, RTime


def FY3D_to_grouplist(sListFile): #time0
    fp = open(sListFile, 'r')
    filelist = [x.rstrip() for x in fp]
    fp.close()
    grouplist = []
    i = 0
    list1 = []
    for j, sLine in enumerate(filelist):

        if j - i >= 1:

            file0 = filelist[j - 1]  # 读取mod03文件
            if j - i == 1:
                list1.append(file0)
            year0 = int(os.path.basename(file0)[19:23])
            mon0 = int(os.path.basename(file0)[23:25])
            day0 = int(os.path.basename(file0)[25:27])
            hour0 = int(os.path.basename(file0)[28:30])
            min0 = int(os.path.basename(file0)[30:32])
            time0 = datetime(year0,mon0,day0,hour0,min0)

            file1 = filelist[j]  # 读取MOD03文件
            year1 = int(os.path.basename(file1)[19:23])
            mon1 = int(os.path.basename(file1)[23:25])
            day1 = int(os.path.basename(file1)[25:27])
            hour1 = int(os.path.basename(file1)[28:30])
            min1 = int(os.path.basename(file1)[30:32])
            time1 = datetime(year1,mon1,day1,hour1,min1)
            delta = (time1 - time0).seconds
            if delta <= 301:  # 间隔小于等于5分钟,300s , 301为留了1s阈值
                list1.append(file1)
                #当j到达最大时，补充最后一组
                if j == len(filelist)-1:
                    grouplist.append(list1)

            else:
                grouplist.append(list1)
                list1 = []
                i = j

    return grouplist


def PixelAgg(RSLib):
    grouplist = FY3D_to_grouplist(sListFile)
    for filelist in grouplist:
        for k, sHdf in enumerate(filelist):
            if 'tmp' in os.path.basename(sHdf):
                continue

            if k ==0:
                SDs = h5py.File(sHdf,'r')  # 加载数据
                LON = SDs['Geolocation/Longitude'][:]
                LON = LON.astype(np.float32)
                LAT = SDs['Geolocation/Latitude'][:]
                LAT = LAT.astype(np.float32)
                nRow, nCol = LON.shape

                sensorZenith = SDs['Geolocation']['SensorZenith']
                VZA = np.where(sensorZenith[:, :] != SDs['Geolocation']['SensorZenith'].attrs['FillValue'][0],
                                        sensorZenith[:, :] * SDs['Geolocation']['SensorZenith'].attrs['Slope'][0] +
                                        SDs['Geolocation']['SensorZenith'].attrs['Intercept'][0], fillvalue1)
                sensorAzimuth = SDs['Geolocation']['SensorAzimuth']
                VAA = np.where(sensorAzimuth[:, :] != SDs['Geolocation']['SensorAzimuth'].attrs['FillValue'][0],
                                         sensorAzimuth[:, :] * SDs['Geolocation']['SensorAzimuth'].attrs['Slope'][0] +
                                         SDs['Geolocation']['SensorAzimuth'].attrs['Intercept'][0], fillvalue1)
                SolarZenith = SDs['Geolocation']['SolarZenith']
                SZA = np.where(SolarZenith[:, :] != SDs['Geolocation']['SolarZenith'].attrs['FillValue'][0],
                               SolarZenith[:, :] * SDs['Geolocation']['SolarZenith'].attrs['Slope'][0] +
                               SDs['Geolocation']['SolarZenith'].attrs['Intercept'][0], fillvalue1)
                SolarAzimuth = SDs['Geolocation']['SolarAzimuth']
                SAA = np.where(SolarAzimuth[:, :] != SDs['Geolocation']['SolarAzimuth'].attrs['FillValue'][0],
                               SolarAzimuth[:, :] * SDs['Geolocation']['SolarAzimuth'].attrs['Slope'][0] +
                               SDs['Geolocation']['SolarAzimuth'].attrs['Intercept'][0], fillvalue1)
                

                FY3D1KM = sHdf.replace('GEO1K','1000M')
                CLMPath = FY3D1KM.replace('FY3D_MERSI_GBAL_L1_','FY3D_MERSI_ORBT_L2_CLM_MLT_NUL_')
                CLMFile = h5py.File(CLMPath, 'r')
                ## 0319CLM数据缺失
                cloud_data_raw = CLMFile['Cloud_Mask'][0,:,:]
                # 第1、2比特表示置信度,00为云, 01为可能云, 10为可能晴空, 11为确信晴空
                bit_data = cloud_data_raw
                bit_0 = bit_data & 0b00000001
                bit_0.astype(int)
                bit1and2 = bit_data & 0b00000110
                #与出来为110，010，100等，除2右移一位转为1确认云，2可能云，3可能晴，4确认晴
                bit1and2 = bit1and2/2+1
                CLM = np.where(bit_0 != 0,bit1and2,0)

                BandRadiance,RTime = read_Radiance_RTime(FY3D1KM)

            else:
                SDs = h5py.File(sHdf,'r')  # 加载数据
                LON1 = SDs['Geolocation/Longitude'][:]
                LON1 = LON1.astype(np.float32)
                LAT1 = SDs['Geolocation/Latitude'][:]
                LAT1 = LAT1.astype(np.float32)
                nRow1, nCol1 = LON1.shape

                sensorZenith1 = SDs['Geolocation']['SensorZenith']
                VZA1 = np.where(sensorZenith1[:, :] != SDs['Geolocation']['SensorZenith'].attrs['FillValue'][0],
                                        sensorZenith1[:, :] * SDs['Geolocation']['SensorZenith'].attrs['Slope'][0] +
                                        SDs['Geolocation']['SensorZenith'].attrs['Intercept'][0], fillvalue1)
                sensorAzimuth1 = SDs['Geolocation']['SensorAzimuth']
                VAA1 = np.where(sensorAzimuth1[:, :] != SDs['Geolocation']['SensorAzimuth'].attrs['FillValue'][0],
                                         sensorAzimuth1[:, :] * SDs['Geolocation']['SensorAzimuth'].attrs['Slope'][0] +
                                         SDs['Geolocation']['SensorAzimuth'].attrs['Intercept'][0], fillvalue1)
                SolarZenith1 = SDs['Geolocation']['SolarZenith']
                SZA1 = np.where(SolarZenith1[:, :] != SDs['Geolocation']['SolarZenith'].attrs['FillValue'][0],
                               SolarZenith1[:, :] * SDs['Geolocation']['SolarZenith'].attrs['Slope'][0] +
                               SDs['Geolocation']['SolarZenith'].attrs['Intercept'][0], fillvalue1)
                SolarAzimuth1 = SDs['Geolocation']['SolarAzimuth']
                SAA1 = np.where(SolarAzimuth1[:, :] != SDs['Geolocation']['SolarAzimuth'].attrs['FillValue'][0],
                               SolarAzimuth1[:, :] * SDs['Geolocation']['SolarAzimuth'].attrs['Slope'][0] +
                               SDs['Geolocation']['SolarAzimuth'].attrs['Intercept'][0], fillvalue1)
                


                FY3D1KM = sHdf.replace('GEO1K','1000M')
                CLMPath1 = FY3D1KM.replace('FY3D_MERSI_GBAL_L1_','FY3D_MERSI_ORBT_L2_CLM_MLT_NUL_')
                CLMFile1 = h5py.File(CLMPath1, 'r')
                cloud_data_raw1 = CLMFile1['Cloud_Mask'][0,:,:]
                # 第1、2比特表示置信度,00为云, 01为可能云, 10为可能晴空, 11为确信晴空
                bit_data_1 = cloud_data_raw1
                bit_0_1 = bit_data_1 & 0b00000001
                bit_0_1.astype(int)
                bit1and2_1 = bit_data_1 & 0b00000110
                #与出来为110，010，100等，除2右移一位转为1确认云，2可能云，3可能晴，4确认晴
                bit1and2_1 = bit1and2_1/2+1
                CLM1 = np.where(bit_0_1 != 0,bit1and2_1,0)
                BandRadiance1,RTime1 = read_Radiance_RTime(FY3D1KM)

                for i in range(len(BandRadiance)):
                    # IN[i] = np.vstack((IN[i], X))

                    BandRadiance[i] = np.vstack((BandRadiance[i],BandRadiance1[i]))

                LON = np.vstack((LON, LON1))
                LAT = np.vstack((LAT, LAT1))
                VZA = np.vstack((VZA, VZA1))
                VAA = np.vstack((VAA, VAA1))
                RTime = np.vstack((RTime, RTime1))
                CLM = np.vstack((CLM,CLM1))
                SZA = np.vstack((SZA, SZA1))
                SAA = np.vstack((SAA, SAA1))

                nRow += nRow1
                print(sHdf+' OK')

        sOutFile = os.path.basename(filelist[k].replace('.HDF', suffix))
        sOutFile = os.path.join(sOutPath ,sOutFile)


        fp = open(sOutFile, 'wb')
        nRow1, nCol1 = LON.shape
            # BandRadiance
        for i in range(len(BandRadiance)):
                bSuccess = RSLib.Granule2GridForFill(np.float32(BandRadiance[i]), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT,
                                              COUNT, COUNTFILL,lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
                fp.write(OUT)

        # Sensor zenith
        # bSuccess = RSLib.Granule2GridforVAA(np.float32(VZA), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT,
        #                                   COUNT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        bSuccess = RSLib.Granule2GridforVAA(VZA.astype(np.float32),np.float32(LON),np.float32(LAT),  nRow1, nCol1, fillvalue1, OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)
        # Sensor azimuth
        bSuccess = RSLib.Granule2GridforVAA(VAA.astype(np.float32),np.float32(LON),np.float32(LAT),  nRow1, nCol1, fillvalue1, OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)

        # RTime
        # bSuccess = RSLib.Granule2Grid(RTime, np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT, lon0, lat0,
        #                                   dxy, nRow2, nCol2, fillvalue2)
        bSuccess = RSLib.Granule2GridforVAA(RTime.astype(np.float32),np.float32(LON),np.float32(LAT),  nRow1, nCol1, fillvalue1, OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)

        fp.write(OUT)
        #CLM
        bSuccess = RSLib.Granule2Grid(CLM.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT, lon0,
                                      lat0,dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)

        # SZA
        # bSuccess = RSLib.Granule2GridforVAA(SZA, np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT,
        #                                   lon0, lat0,dxy, nRow2, nCol2, fillvalue2)
        bSuccess = RSLib.Granule2GridforVAA(SZA.astype(np.float32),np.float32(LON),np.float32(LAT),  nRow1, nCol1, fillvalue1, OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)
            # SAA
        # bSuccess = RSLib.Granule2Grid(SAA, np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT,
        #                                   lon0, lat0,dxy, nRow2, nCol2, fillvalue2)
        bSuccess = RSLib.Granule2GridforVAA(SAA.astype(np.float32),np.float32(LON),np.float32(LAT),  nRow1, nCol1, fillvalue1, OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)

        fp.close()


        sHdrFile = sOutFile.replace('.dat', '.hdr')
        fp = open(sHdrFile, 'w')
        fp.write('ENVI\ndescription = {\n  File Imported into ENVI.}\n')
        fp.write('samples = {0:d}\nlines   = {1:d}\nbands   = {2:d}\n'.format(nCol2, nRow2, 25))
        fp.write(
                'header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\nsensor type = Unknown\nbyte order = 0\nwavelength units = Unknown\n')
        fp.write('band names = {B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,VZA,VAA,RTime,CLM,SZA,SAA}')  # FY-3E MERSI
            # fp.write('band names = {CH24, CH25, VZA, VAA, RTime}') # FY-3D MERSI
        fp.close()

        # np.save(fp,OUT)
        print(sHdf+'-Finished!')

if __name__ == '__main__':
    RSLib = RSLibImport(r'C:\Users\1\Desktop\py\Match\RSLib_test.so')
    PixelAgg(RSLib)

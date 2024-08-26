import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import h5py

Month = 'June'
# 数据列表路径并读取
sListFile = r'D:/VIIRS/'+Month+'/VJ103List.dat'
sOutPath = r'D:/VIIRS/'+Month+'/SampleData'
# 初始化信息

nRow2 = 721
nCol2 = 1440
lon0 = 0.0
lat0 = 90.0
dxy = 0.25
suffix = '_0.25X0.25.dat'
fillvalue1 = 0
fillvalue2 = 0

mtime0 = datetime(1993, 1, 1, 0, 0)
time0 = datetime(2021, 1, 1, 0, 0)


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


def findVJ102(keyword, path):
    file = []
    files = os.listdir(path)
    for i in files:
        i = os.path.join(path, i)  # 合并路径与文件名
        # if os.path.isfile(i):#判断是否为文件
        # print(i)
        if keyword in os.path.basename(i):
            print('Find Match ' + keyword)  # 绝对路径
            file = i
            break
    if file == []:
        print('No Match File with MYD021KM ' + keyword)
    return file

def findCLDMSK(keyword, path):
    file = []



    files = os.listdir(path)
    for i in files:
        i = os.path.join(path, i)  # 合并路径与文件名
        # if os.path.isfile(i):#判断是否为文件
        # print(i)
        if keyword in os.path.basename(i):
            # print('Find Match MYD021KM ' + keyword)  # 绝对路径
            file = i
            break
    if file == []:
        print('No Match File with MYD35 ' + keyword)
    return file


def readVJ102_Radiance(VJ102path):
    # 读取VJ102
    VJ102 = h5py.File(VJ102path)
    DN = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    # for i in range(1,14,1):
    #     if i <= 9:
    #         DN[i] = VJ102['observation_data'][f'M0{i}'][:]
    #     if i >=10 :
    #         DN[i] = VJ102['observation_data'][f'M{i}']

    BandRadiance = [1,2,3,4,5,6,7,8,9,10,11,12,13]

    for i in range(1,14,1):
        if i <= 9:
            BandRadiance[i-1] = np.where(VJ102['observation_data'][f'M0{i}'][:] <= 65527, 
                                       VJ102['observation_data'][f'M0{i}'][:] * VJ102['observation_data'][f'M0{i}'].attrs['radiance_scale_factor']
                                       +VJ102['observation_data'][f'M0{i}'].attrs['radiance_add_offset'],fillvalue1)

        if i >=10 :
            if i ==12:
                scale_factor = 7.63044e-005
                offset = 0.000834405
            elif i ==13 :
                scale_factor = 0.00161748
                offset = 0.00108056
            else:
                scale_factor = VJ102['observation_data'][f'M{i}'].attrs['radiance_scale_factor']
                offset = VJ102['observation_data'][f'M{i}'].attrs['radiance_add_offset']


            BandRadiance[i-1] = np.where(VJ102['observation_data'][f'M{i}'][:] <= 65527, 
                                       VJ102['observation_data'][f'M{i}'][:] *scale_factor
                                       +offset,fillvalue1)


    return BandRadiance

def to_grouplist(sListFile): #time0
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
            day0 = float(os.path.basename(file0)[14:17]) - 1  # 差一天
            hour0 = float(os.path.basename(file0)[18:20])
            min0 = float(os.path.basename(file0)[20:22])
            time0 = timedelta(days=day0, hours=hour0, minutes=min0)

            file1 = filelist[j]  # 读取MOD03文件
            Day1 = float(os.path.basename(file1)[14:17]) - 1
            Hour1 = float(os.path.basename(file1)[18:20])
            min1 = float(os.path.basename(file1)[20:22])
            time1 = timedelta(days=Day1, hours=Hour1, minutes=min1)
            delta = (time1 - time0).seconds
            # 6分钟间隔
            if delta <= 361:  # 间隔小于等于5分钟,300s , 301为留了1s阈值
                list1.append(file1)
                # 当j到达最大时，补充最后一组
                if j == len(filelist)-1:
                    grouplist.append(list1)
            else:
                grouplist.append(list1)
                list1 = []
                i = j

    return grouplist



def PixelAgg(RSLib):

    grouplist = to_grouplist(sListFile)

    for filelist in grouplist:

        for k, sNC in enumerate(filelist):


            if k ==0:
                # print(sHdf, end='...')
                SDs = h5py.File(sNC)
                LON = SDs['geolocation_data']['longitude'][:]
                LAT = SDs['geolocation_data']['latitude'][:]
                OUT = np.zeros((nRow2, nCol2), dtype=np.float32)
                COUNT = np.zeros((nRow2, nCol2), dtype=np.float32)
                COUNTFILL = np.zeros((nRow2, nCol2), dtype=np.float32)
                VZA = SDs['geolocation_data']['sensor_zenith'][:]
                VZA = np.where(VZA != SDs['geolocation_data']['sensor_zenith'].attrs['_FillValue'], VZA * 0.01,fillvalue1)
                VAA = SDs['geolocation_data']['sensor_azimuth'][:]
                VAA = np.where(VAA != SDs['geolocation_data']['sensor_azimuth'].attrs['_FillValue'], VAA * 0.01,fillvalue1)
                SZA = SDs['geolocation_data']['solar_zenith'][:]
                SZA = np.where(SZA != SDs['geolocation_data']['solar_zenith'].attrs['_FillValue'], SZA * 0.01,fillvalue1)
                SAA = SDs['geolocation_data']['solar_azimuth'][:]
                SAA = np.where(SAA != SDs['geolocation_data']['solar_azimuth'].attrs['_FillValue'], SAA * 0.01,fillvalue1)
                nRow, nCol = LON.shape
                RTime = np.zeros((nRow, nCol), dtype=np.float32)
                StartTimestr = str(SDs.attrs['StartTime'])
                EndTime = str(SDs.attrs['EndTime'])
                StartTimestr = StartTimestr[2:21]
                EndTime = EndTime[2:21]
                # StartTimestr2 = StartTimestr[13:21]
                StartTimeDate = datetime.strptime(StartTimestr,  "%Y-%m-%d %H:%M:%S")
                StartSeconds = (StartTimeDate-time0).total_seconds()
                EndTimeDate = datetime.strptime(EndTime,  "%Y-%m-%d %H:%M:%S")
                EndTimeSeconds = (EndTimeDate-time0).total_seconds()
                linescantime = (EndTimeSeconds - StartSeconds) / nRow
                for i in range(0, nRow):
                    dt = timedelta(days=float((StartSeconds+(linescantime*i)) / 86400))
                    RTime[i] = np.float32(dt.days + (dt.seconds + dt.microseconds / 1000000) / 86400.0)  # in days
            #CLM
                # 读取MOD03文件
                file = filelist[k]
                # 读取MOD03文件 时间关键字
                Keyword = os.path.basename(file)[9:22]
                CLMpath = findCLDMSK('CLDMSK_L2_VIIRS_NOAA20.' + Keyword, os.path.dirname(file))
                CLMdata = h5py.File(CLMpath)
                cloud_data_raw = CLMdata['geophysical_data']['Cloud_Mask']
                #数据 比特0表示为否进行了检测, determined or not
                # 第1、2比特表示置信度,00为云, 01为可能云, 10为可能晴空, 11为确信晴空
                bit_data = cloud_data_raw[0, :, :]
                bit_0 = bit_data & 0b00000001
                bit_0.astype(int)
                bit1and2 = bit_data & 0b00000110
                #与出来为110，010，100等，除2右移一位转为0确认云，1可能云，2可能晴，3确认晴
                bit1and2 = bit1and2/2 + 1
                CLM = np.where(bit_0 != 0,bit1and2, 0)


            # 获取MOD201KM数据


                VJ102path = findVJ102('VJ102MOD.' + Keyword, os.path.dirname(file))  # 找到文件名
                # 如果找不到MOD03和MOD01匹配的文件就跳出循环
                if VJ102path == []:
                    continue
                # DN转辐亮度
                BandRadiance = readVJ102_Radiance(VJ102path)

            else :

                SDs = h5py.File(sNC)
                LON1 = SDs['geolocation_data']['longitude'][:]
                LAT1 = SDs['geolocation_data']['latitude'][:]
                VZA1 = SDs['geolocation_data']['sensor_zenith'][:]
                VZA1 = np.where(VZA1 != SDs['geolocation_data']['sensor_zenith'].attrs['_FillValue'][0], VZA1 * 0.01,fillvalue1)
                VAA1 = SDs['geolocation_data']['sensor_azimuth'][:]
                VAA1 = np.where(VAA1 != SDs['geolocation_data']['sensor_azimuth'].attrs['_FillValue'][0], VAA1 * 0.01,fillvalue1)
                SZA1 = SDs['geolocation_data']['solar_zenith'][:]
                SZA1 = np.where(SZA1 != SDs['geolocation_data']['solar_zenith'].attrs['_FillValue'][0], SZA1 * 0.01,fillvalue1)
                SAA1 = SDs['geolocation_data']['solar_azimuth'][:]
                SAA1 = np.where(SAA1 != SDs['geolocation_data']['solar_azimuth'].attrs['_FillValue'][0], SAA1 * 0.01,fillvalue1)
                nRow1, nCol1 = LON1.shape
                RTime1 = np.zeros((nRow1, nCol1), dtype=np.float32)
                StartTimestr1 = str(SDs.attrs['StartTime'])
                EndTime1 = str(SDs.attrs['EndTime'])
                StartTimestr1 = StartTimestr1[2:21]
                EndTime1 = EndTime1[2:21]
                # StartTimestr2 = StartTimestr[13:21]
                StartTimeDate1 = datetime.strptime(StartTimestr1,  "%Y-%m-%d %H:%M:%S")
                StartSeconds1 = (StartTimeDate1-time0).total_seconds()
                EndTimeDate1 = datetime.strptime(EndTime1,  "%Y-%m-%d %H:%M:%S")
                EndTimeSeconds1 = (EndTimeDate1-time0).total_seconds()
                linescantime1 = (EndTimeSeconds1 - StartSeconds1) / nRow1
                for i in range(0, nRow1):
                    dt = timedelta(days=float((StartSeconds1+(linescantime1*i)) / 86400))
                    RTime1[i] = np.float32(dt.days + (dt.seconds + dt.microseconds / 1000000) / 86400.0)  # in days
            
                # 读取MOD03文件
                file = filelist[k]
                # 读取MOD03文件 时间关键字
                Keyword = os.path.basename(file)[9:22]
                MYD35path = findCLDMSK('CLDMSK_L2_VIIRS_NOAA20.' + Keyword, os.path.dirname(file))
                CLMdata = h5py.File(MYD35path)
                cloud_data_raw = CLMdata['geophysical_data']['Cloud_Mask']
                #数据 比特0表示为否进行了检测, determined or not
                # 第1、2比特表示置信度,00为云, 01为可能云, 10为可能晴空, 11为确信晴空
                bit_data = cloud_data_raw[0, :, :]
                bit_0 = bit_data & 0b00000001
                bit_0.astype(int)
                bit1and2 = bit_data & 0b00000110
                #与出来为110，010，100等，除2右移一位转为0确认云，1可能云，2可能晴，3确认晴
                bit1and2 = bit1and2/2 + 1
                CLM1 = np.where(bit_0 != 0,bit1and2,0)

                # 获取MOD201KM数据
                VJ102path = findVJ102('VJ102MOD.' + Keyword, os.path.dirname(file))  # 找到文件名
                # 如果找不到MOD03和MOD01匹配的文件就跳出循环
                if VJ102path == []:
                        continue
                # DN转辐亮度
                BandRadiance1 = readVJ102_Radiance(VJ102path)

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






        sOutFile = os.path.basename(filelist[k].replace('.nc', suffix))
        sOutFile = sOutFile.replace('VJ103','VJ102')
        ## 是 ** = **.replace不然不管用
        sOutFile = os.path.join(sOutPath ,sOutFile)
        

        fp = open(sOutFile, 'wb')
        nRow1, nCol1 = LON.shape
        # BandRadiance
        for i in range(len(BandRadiance)):
            bSuccess = RSLib.Granule2GridForFill(np.float32(BandRadiance[i]), np.float32(LON), np.float32(LAT), nRow1,nCol1, fillvalue1, OUT,COUNT,COUNTFILL, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)


            fp.write(OUT)

        # Sensor zenith
        bSuccess = RSLib.Granule2GridforVAA(VZA.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1,
                                                OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        
        # bSuccess = RSLib.Granule2Grid(np.float32(VZA), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT,
        #                                   COUNT,lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)
        # Sensor azimuth
        bSuccess = RSLib.Granule2GridforVAA(VAA.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1,
                                                OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)

        # RTime
        # bSuccess = RSLib.Granule2Grid(RTime, np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT, lon0, lat0,
        #                                   dxy, nRow2, nCol2, fillvalue2)
        bSuccess = RSLib.Granule2GridforVAA(RTime.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1,
                                                OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)

        #CLM
        bSuccess = RSLib.Granule2Grid(CLM.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT, lon0,
                                      lat0,dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)

        #SZA
        bSuccess = RSLib.Granule2GridforVAA(SZA.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1,
                                                OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        # bSuccess = RSLib.Granule2Grid(SZA.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT, lon0,
        #                               lat0,dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)

        #SAA
        bSuccess = RSLib.Granule2GridforVAA(SAA.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1,
                                                OUT, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)
        # bSuccess = RSLib.Granule2Grid(SAA.astype(np.float32), np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT, lon0,
        #                               lat0,dxy, nRow2, nCol2, fillvalue2)
        fp.write(OUT)

        fp.close()

        sHdrFile = sOutFile.replace('.dat', '.hdr')
        fp = open(sHdrFile, 'w')
        fp.write('ENVI\ndescription = {\n  File Imported into ENVI.}\n')
        fp.write('samples = {0:d}\nlines   = {1:d}\nbands   = {2:d}\n'.format(nCol2, nRow2, 19))
        fp.write(
                'header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\ndata ignore value = 0\n'
                'sensor type = Unknown\nbyte order = 0\nwavelength units = Unknown\n')
        # # x and y start from zero
        # 'x start = {{0:d}}\ny start = {{1:d}}\n'.format((int)(lon0 / dxy + 720),
                                                        # (int)((90 - lat0) / dxy))
        # fp.write(
        #         'map info = {{Geographic Lat/Lon, 1.0000, 1.0000, {0:d}, {1:d}, 2.500000e-01, 2.500000e-01, WGS-84, units=Degrees}}\n'.format(
        #             lon0, lat0))  # 
        fp.write(
                'band names = {B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,VZA,VAA,RTime,CLM,SZA,SAA}')  # FY-3E MERSI
            # fp.write('band names = {CH24, CH25, VZA, VAA, RTime}') #
        fp.close()
        print('Batch_Finished!\n')


if __name__ == '__main__':
    RSLib = RSLibImport(r'C:\Users\1\Desktop\py\Match\RSLib_test.so')
    PixelAgg(RSLib)


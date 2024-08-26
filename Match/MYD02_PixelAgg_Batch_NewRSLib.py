import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import glob

Month = 'December'
# 数据列表路径并读取
sListFile = r'D:/MODIS/'+Month+'/MYD03List.dat'
GEOpath = r'D:/MODIS/'+Month+'/1KM&GEO&CLM'
list1 = open(sListFile, 'w')
geofile = glob.glob(GEOpath +'/MYD03*.hdf')
list1.write('\n'.join(geofile))
list1.close()
sListFile = r'D:/MODIS/'+Month+'/MYD03List.dat'

sOutPath = r'C:/Users/1/Desktop/casestudy/data/MODIS/'+Month+'/SampleData'
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


def findmyd02(keyword, path):
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

def findmyd35(keyword, path):
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


def readMYD021KM_Radiance(MOD021KMpath):
    # 读取MOD021KM
    MOD021KM = HDF.SD(MOD021KMpath)
    RefSB1KM = MOD021KM.select('EV_1KM_RefSB').get()
    ##  原数据为int16，转为float32
    RefSB1KM = np.float32(RefSB1KM)
    # 存放通道数据的矩阵

    EV_refSB1KM_Rad_Scale = np.array(MOD021KM.select('EV_1KM_RefSB').attributes(full=1).get('radiance_scales')[0])
    EV_refSB1KM_Rad_offsets = np.array(MOD021KM.select('EV_1KM_RefSB').attributes(full=1).get('radiance_offsets')[0])

    RefSB500 = MOD021KM.select('EV_500_Aggr1km_RefSB').get()
    RefSB500 = np.float32(RefSB500)

    EV_refSB500_Rad_Scale = np.array(
        MOD021KM.select('EV_500_Aggr1km_RefSB').attributes(full=1).get('radiance_scales')[0])
    EV_refSB500_Rad_offsets = np.array(
        MOD021KM.select('EV_500_Aggr1km_RefSB').attributes(full=1).get('radiance_offsets')[0])

    RefSB250m = MOD021KM.select('EV_250_Aggr1km_RefSB').get()
    RefSB250m = np.float32(RefSB250m)
    EV_refSB250_Rad_Scale = np.array(
        MOD021KM.select('EV_250_Aggr1km_RefSB').attributes(full=1).get('radiance_scales')[0])
    EV_refSB250_Rad_offsets = np.array(
        MOD021KM.select('EV_250_Aggr1km_RefSB').attributes(full=1).get('radiance_offsets')[0])

    BandRadiance = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    for i in range(21):
        if i <= 1:
            RefSB250m[i] = np.where((0<RefSB250m[i, :, :] )&(RefSB250m[i, :, :] < 32000),
                                          (RefSB250m[i, :, :] - EV_refSB250_Rad_offsets[i]) * EV_refSB250_Rad_Scale[i],
                                          fillvalue1)
            BandRadiance[i] = RefSB250m[i]
        if 1 < i <= 6:
            q = i-2
            RefSB500[q, :, :] = np.where((0<RefSB500[q, :, :])&(RefSB500[q, :, :] < 32000),
                                        (RefSB500[q, :, :] - EV_refSB500_Rad_offsets[q]) *EV_refSB500_Rad_Scale[q], 
                                        fillvalue1)
            BandRadiance[i] = RefSB500[q]
        if 6 < i <= 20: 
            k = i- 7
            RefSB1KM[k, :, :]  = np.where((0<RefSB1KM[k, :, :])&(RefSB1KM[k, :, :] < 32000),
                                             (RefSB1KM[k, :, :] - EV_refSB1KM_Rad_offsets[k]) *
                                             EV_refSB1KM_Rad_Scale[k], fillvalue1)
            BandRadiance[i] = RefSB1KM[k]

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
            day0 = float(os.path.basename(file0)[11:14]) - 1  # 差一天
            hour0 = float(os.path.basename(file0)[15:17])
            min0 = float(os.path.basename(file0)[17:19])
            time0 = timedelta(days=day0, hours=hour0, minutes=min0)

            file1 = filelist[j]  # 读取MOD03文件
            Day1 = float(os.path.basename(file1)[11:14]) - 1
            Hour1 = float(os.path.basename(file1)[15:17])
            min1 = float(os.path.basename(file1)[17:19])
            time1 = timedelta(days=Day1, hours=Hour1, minutes=min1)
            delta = (time1 - time0).seconds
            if delta <= 301:  # 间隔小于等于5分钟,300s , 301为留了1s阈值
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

        for k, sHdf in enumerate(filelist):


            if k ==0:
                # print(sHdf, end='...')
                SDs = HDF.SD(sHdf)
                LON = SDs.select('Longitude').get()
                LAT = SDs.select('Latitude').get()
                OUT = np.zeros((nRow2, nCol2), dtype=np.float32)
                COUNT = np.zeros((nRow2, nCol2), dtype=np.float32)
                COUNTFILL = np.zeros((nRow2, nCol2), dtype=np.float32)
                VZA = SDs.select('SensorZenith').get()
                VZA = np.where(VZA != SDs.select('SensorZenith').attributes(full=1).get('_FillValue')[0], VZA * 0.01,fillvalue1)
                VAA = SDs.select('SensorAzimuth').get()
                VAA = np.where(VAA != SDs.select('SensorAzimuth').attributes(full=1).get('_FillValue')[0], VAA * 0.01,fillvalue1)
                SZA = SDs.select('SolarZenith').get()
                SZA = np.where(SZA != SDs.select('SolarZenith').attributes(full=1).get('_FillValue')[0], SZA * 0.01,fillvalue1)
                SAA = SDs.select('SolarAzimuth').get()
                SAA = np.where(SAA != SDs.select('SolarAzimuth').attributes(full=1).get('_FillValue')[0], SAA * 0.01,fillvalue1)
                nRow, nCol = LON.shape
                RTime = np.zeros((nRow, nCol), dtype=np.float32)
                EVscantime = SDs.select('EV start time').get()
                Startscantime = EVscantime[0]
                Endingscantime = Startscantime+ 299.9
                linescantime = (Endingscantime -Startscantime) / nRow
                for i in range(0, nRow):
                    dt = mtime0 + timedelta(days=float((Startscantime+(linescantime*i)) / 86400)) - time0
                    RTime[i] = np.float32(dt.days + (dt.seconds + dt.microseconds / 1000000) / 86400.0)  # in days
            #CLM
                # 读取MOD03文件
                file = filelist[k]
                # 读取MOD03文件 时间关键字
                Keyword = os.path.basename(file)[5:19]
                MYD35path = findmyd35('MYD35_L2' + Keyword, os.path.dirname(file))
                CLMdata = HDF.SD(MYD35path)
                cloud_data_raw = CLMdata.select('Cloud_Mask').get()
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


                MYD021KMpath = findmyd02('MYD021KM' + Keyword, os.path.dirname(file))  # 找到文件名
                # 如果找不到MOD03和MOD01匹配的文件就跳出循环
                if MYD021KMpath == []:
                    continue
                # DN转辐亮度
                BandRadiance = readMYD021KM_Radiance(MYD021KMpath)

            else :

                SDs = HDF.SD(sHdf)
                LON1 = SDs.select('Longitude').get()
                LAT1 = SDs.select('Latitude').get()
                VZA1 = SDs.select('SensorZenith').get()
                VZA1 = np.where(VZA1 != SDs.select('SensorZenith').attributes(full=1).get('_FillValue')[0], VZA1 * 0.01,fillvalue1)
                VAA1 = SDs.select('SensorAzimuth').get()
                VAA1 = np.where(VAA1 != SDs.select('SensorAzimuth').attributes(full=1).get('_FillValue')[0], VAA1 * 0.01,fillvalue1)
                SZA1 = SDs.select('SolarZenith').get()
                SZA1 = np.where(SZA1 != SDs.select('SolarZenith').attributes(full=1).get('_FillValue')[0], SZA1 * 0.01,fillvalue1)
                SAA1 = SDs.select('SolarAzimuth').get()
                SAA1 = np.where(SAA1 != SDs.select('SolarAzimuth').attributes(full=1).get('_FillValue')[0], SAA1 * 0.01,fillvalue1)
                nRow1, nCol1 = LON1.shape
                RTime1 = np.zeros((nRow1, nCol1), dtype=np.float32)
                EVscantime = SDs.select('EV start time').get()
                Startscantime = EVscantime[0]
                Endingscantime = Startscantime +299.9
                linescantime = (Endingscantime - Startscantime) / nRow1
                for i in range(0, nRow1):
                    dt = mtime0 + timedelta(days=float((Startscantime + (linescantime * i)) / 86400)) - time0
                    RTime1[i] = np.float32(dt.days + (dt.seconds + dt.microseconds / 1000000) / 86400.0)  # in days

                # 读取MOD03文件
                file = filelist[k]
                # 读取MOD03文件 时间关键字
                Keyword = os.path.basename(file)[5:19]
                MYD35path = findmyd35('MYD35_L2' + Keyword, os.path.dirname(file))
                CLMdata = HDF.SD(MYD35path)
                cloud_data_raw = CLMdata.select('Cloud_Mask').get()
                #数据 比特0表示为否进行了检测, determined or not
                # 第1、2比特表示置信度,00为云, 01为可能云, 10为可能晴空, 11为确信晴空
                bit_data = cloud_data_raw[0, :, :]
                bit_0 = bit_data & 0b00000001
                bit_0.astype(int)
                bit1and2 = bit_data & 0b00000110
                #与出来为110，010，100等，除2右移一位转为1确认云，2可能云，3可能晴，4确认晴
                bit1and2 = bit1and2/2+1
                CLM1 = np.where(bit_0 != 0,bit1and2,0)

                # 获取MOD201KM数据
                MYD021KMpath = findmyd02('MYD021KM' + Keyword, os.path.dirname(file))  # 找到文件名
                # 如果找不到MOD03和MOD01匹配的文件就跳出循环
                if MYD021KMpath == []:
                        continue
                # DN转辐亮度
                BandRadiance1 = readMYD021KM_Radiance(MYD021KMpath)

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






        sOutFile = os.path.basename(filelist[k].replace('.hdf', suffix))
        sOutFile = sOutFile.replace('MYD03','MYD02')
        ## 是 ** = **.replace不然不管用
        sOutFile = os.path.join(sOutPath ,sOutFile)
        

        fp = open(sOutFile, 'wb')
        nRow1, nCol1 = LON.shape
        # BandRadiance
        for i in range(len(BandRadiance)):
            bSuccess = RSLib.Granule2GridForFill(np.float32(BandRadiance[i]), np.float32(LON), np.float32(LAT), nRow1,nCol1, fillvalue1, OUT,COUNT,COUNTFILL, lon0, lat0, dxy, nRow2, nCol2, fillvalue2)

            # if sum(sum(OUT[:, :] > 0)) / (nRow2 * nCol2) < 0.001:
            #     fp.close()
            #     os.remove(sOutFile)
            #     print('Data not stored.\n')
            #     continue

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
        # bSuccess = RSLib.Granule2GridforVAA(RTime, np.float32(LON), np.float32(LAT), nRow1, nCol1, fillvalue1, OUT, COUNT, lon0, lat0,
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
        fp.write('samples = {0:d}\nlines   = {1:d}\nbands   = {2:d}\n'.format(nCol2, nRow2, 27))
        fp.write(
                'header offset = 0\nfile type = ENVI Standard\ndata type = 4\ninterleave = bsq\ndata ignore value = 0\n'
                'sensor type = Unknown\nbyte order = 0\nwavelength units = Unknown\n')
        # # x and y start from zero
        # 'x start = {{0:d}}\ny start = {{1:d}}\n'.format((int)(lon0 / dxy + 720),
                                                        # (int)((90 - lat0) / dxy))
        # fp.write(
        #         'map info = {{Geographic Lat/Lon, 1.0000, 1.0000, {0:d}, {1:d}, 2.500000e-01, 2.500000e-01, WGS-84, units=Degrees}}\n'.format(
        #             lon0, lat0))  # 非占位符{}要双写
        fp.write(
                'band names = {B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13L,B13H,B14L,B14H,B15,B16,B17,B18,B19,VZA,VAA,RTime,CLM,SZA,SAA}')  # FY-3E MERSI
            # fp.write('band names = {CH24, CH25, VZA, VAA, RTime}') # FY-3D MERSI
        fp.close()
        print('Batch_Finished!\n')


if __name__ == '__main__':
    RSLib = RSLibImport(r'C:\Users\1\Desktop\py\Match\RSLib_test.so')
    PixelAgg(RSLib)


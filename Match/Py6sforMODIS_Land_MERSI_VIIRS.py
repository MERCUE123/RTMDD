# -*- coding: utf-8 -*-
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
import h5py as h5py

KEY = 'MERSI'
match_sensor = 'MERSI_VIIRS'
Bias = 0
dTime = 900
Month = 'March'
match match_sensor:
    case 'MERSI_VIIRS':
        sOutFile = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/Py6s_'+KEY+'_'+Month+'_With_VIIRS.csv'
        MatchTablePath = r'C:\Users\1\Desktop\casestudy\data\MatchRes/'+Month+'/LandMatchTable_'+match_sensor+'_'+Month+'.csv'
# MatchTablePath = r"C:\Users\1\Desktop\casestudy\data\MatchRes\March\LandMatchTable_MERSI_AQUA_Newlib3.csv"
BRDFpath = r'C:\Users\1\Desktop\casestudy\data\MODIS/'+Month+'/BRDF/BRDFC1_Sampleto_19CH_'+KEY
AeroSamplepath = r'C:\Users\1\Desktop\casestudy\data\ERA5/Aero/'+Month+'/ampleAeroGrid2' #.dat
Aeroncpath = r'C:\Users\1\Desktop\casestudy\data\ERA5/Aero/'+Month+'/ERA5Aero'
AeroMODIS_Samplepath = r'C:\Users\1\Desktop\casestudy\data\ERA5/Aero/'+Month+'/MODISAero_Sampleto_0.25x0.25'
HeightPath = r"C:\Users\1\Desktop\casestudy\data\landfile\GlobalDEM-GOTOP30_0.25X0.25.dat"



MODIS = [PredefinedWavelengths.ACCURATE_MODIS_AQUA_1,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_2,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_3,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_4,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_5,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_6,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_7,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_8,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_9,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_10,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_11,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_12,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_13,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_14,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_15,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_16,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_17,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_18,
PredefinedWavelengths.ACCURATE_MODIS_AQUA_19,
        ]
MERSI= [PredefinedWavelengths.ACCURATE_MERSI_1,
PredefinedWavelengths.ACCURATE_MERSI_2,
PredefinedWavelengths.ACCURATE_MERSI_3,
PredefinedWavelengths.ACCURATE_MERSI_4,
PredefinedWavelengths.ACCURATE_MERSI_5,
PredefinedWavelengths.ACCURATE_MERSI_6,
PredefinedWavelengths.ACCURATE_MERSI_7,
PredefinedWavelengths.ACCURATE_MERSI_8,
PredefinedWavelengths.ACCURATE_MERSI_9,
PredefinedWavelengths.ACCURATE_MERSI_10,
PredefinedWavelengths.ACCURATE_MERSI_11,
PredefinedWavelengths.ACCURATE_MERSI_12,
PredefinedWavelengths.ACCURATE_MERSI_13,
PredefinedWavelengths.ACCURATE_MERSI_14,
PredefinedWavelengths.ACCURATE_MERSI_15,
PredefinedWavelengths.ACCURATE_MERSI_16,
PredefinedWavelengths.ACCURATE_MERSI_17,
PredefinedWavelengths.ACCURATE_MERSI_18,
PredefinedWavelengths.ACCURATE_MERSI_19
]

def get_atmos(ERA5time,ilat,jlon,HeightTarget):



    ## 取原数据
    
    era5dir = r'C:\Users\1\Desktop\casestudy\data\ERA5/atmos/'+Month
    era5path = os.path.join(era5dir, '{0}{1}{2}.nc'.format(str(ERA5time.year),str(ERA5time.month).zfill(2),str(ERA5time.day).zfill(2)))
    atmos = Dataset(era5path)
    #具体时间

    #数据提取
    Pressure = atmos.variables['level'][:].data
    Pressure = Pressure[0:37]

    # 取层
    Q = list(np.arange(5))

        # 取层
    Q[0] = atmos.variables['o3'][ERA5time.hour, 0:37, ilat, jlon].data
    Q[1] = atmos.variables['t'][ERA5time.hour, 0:37, ilat, jlon].data
    Q[2] = atmos.variables['r'][ERA5time.hour, 0:37, ilat, jlon].data
    Q[3] = atmos.variables['z'][ERA5time.hour, 0:37, ilat, jlon].data
    # Q[4] = atmos.variables['q'][ERA5time.hour, 0:37, ilat, jlon].data
    # 判断相对湿度
    RH = np.max(Q[2][0:15])
    # 臭氧单位转换Kg/Kg to g/m3
    # rou(kg/m3) = alpha(kg/kg)*P(Pa)/R (R = R0*M = 8.314*相对分子质量)*T(K)
    # PV = NRT
    R0 = 8.314


    #臭氧
    Q[0] = Q[0] * (Pressure * 100)*48 / (R0 * Q[1]) 
    
    #采用绝对湿度计算
    # 水气单位转换Kg/Kg to g/m3
    # Q[4] = Q[4] * (Pressure * 100)
    # Q[4] = Q[4] *18 / (R0 * Q[1])
    # AbsWater = Q[4] *18 / (R0 * Q[1])

    # .用修正的Tetens 公式 -30~30摄氏度 

    # TetensSaturWaterPressure = 6.112 * np.exp((17.62 * (Q[1]- 273.15)) / ((Q[1]- 273.15) + 243.12))*100
    # TetensRealtiveWater= TetensSaturWaterPressure*Q[2] *18 /(R0*Q[1])
    # Q[4] = TetensRealtiveWater

    # .用BUCK公式：
    t = Q[1]-273.15
    NegtiveSaturWaterPressure = 6.1115*np.exp((23.036-t/333.7)*t/(279.82+t))*100    
    PositiveSaturWaterPressure = 6.1121*np.exp((18.678-t/234.5)*t/(257.14+t))*100
    BUCKSaturWaterPressure = np.where(t<0,NegtiveSaturWaterPressure,PositiveSaturWaterPressure)
    BUCKRealtiveWater= BUCKSaturWaterPressure*Q[2]/100
    WaterDensity = BUCKRealtiveWater*18/(R0*Q[1])
    Q[4] = WaterDensity
    # 位势高度转换km
    lat = ilat
    g0=9.80616*(1 - 0.0026373*(np.cos(2*(np.pi)*lat/180.0) + 0.0000059*np.cos(2*(np.pi)*lat/180.0)*np.cos(2*(np.pi)*lat/180.0)))
    # g0 = (float)(9806.16*(1 - 0.0026373*cos(2*PI*lat/180.0) + 0.0000059*cos(2*PI*lat/180.0)*cos(2*PI*lat/180.0)));
    
    Q[3] = Q[3] / g0 / 1000
    # 气压帕斯卡转为毫巴
    # Pressure = Pressure
    Y = [Q[3], Pressure, Q[1], Q[4], Q[0]]

    atmos = {
        "altitude": Y[0][::-1],
        "pressure": Y[1][::-1],
        "temperature": Y[2][::-1],
        "water": Y[3][::-1],
        "ozone": Y[4][::-1]
    }
    ## 插值
    #取数据
    altitude = atmos['altitude']
    pressure = atmos['pressure']
    temperature = atmos['temperature']
    water = atmos['water'] 
    ozone = atmos['ozone'] 
    #如果目标海拔小于altitude的最小值

    if HeightTarget <= altitude[0]:
        atmos['altitude'] = atmos['altitude'][0:34]
        atmos['pressure'] = atmos['pressure'][0:34]
        atmos['temperature'] = atmos['temperature'][0:34]
        atmos['water'] = atmos['water'][0:34]
        atmos['ozone'] = atmos['ozone'][0:34]

    #取出地表以上的数据
    if HeightTarget > altitude[0]:
        new_altitude = altitude[altitude > HeightTarget]



        #插值海拔
        # 如果altitude最小值高于地表，且层数多于34层
        if len(new_altitude) >=33:
            #恰好加地面34层
            if len(new_altitude) ==33:
                interpolated_altitude =np.append(HeightTarget,new_altitude)

            #加地面超过34层
            if len(new_altitude)>33:
                interpolated_altitude =np.append(HeightTarget,new_altitude[1:34])
            # interpolated_altitude = np.linspace(HeightTarget, max_altitude, 34)
            

            #对其他数据插值，1阶插值
            altitude_temperature = interpolate.UnivariateSpline(altitude, temperature,s=0,k=1)
            interpolated_temperature = altitude_temperature(interpolated_altitude)
            altitude_water = interpolate.UnivariateSpline(altitude, water,s=0,k=1)
            interpolated_water= altitude_water(interpolated_altitude)
            np.where(interpolated_water<0,0,interpolated_water)
            altitude_ozone = interpolate.UnivariateSpline(altitude, ozone,s=0,k=1)
            interpolated_ozone = altitude_ozone(interpolated_altitude)
            np.where(interpolated_ozone<0,0,interpolated_ozone)
            altitude_pressure = interpolate.UnivariateSpline(altitude, pressure,s=0,k=1)
            interpolated_pressure = altitude_pressure(interpolated_altitude)
            np.where(interpolated_pressure<0,0,interpolated_pressure)
            
        # 如果altitude最小值高于地表，且层数小于34层
        if len(new_altitude) <33:
            interpolated_altitude =np.append(HeightTarget,new_altitude)

        # interpolated_altitude = interpolated_altitude[5:37]
            # 加地面缺k层
            k = 34 - len(interpolated_altitude)

            # 创建一个新的数组，从0到k+1，间隔为1
            line = np.arange(k+1)
            # 从0到k+1进行拟合，间隔为1，确保ERA5原数据不变
            interpolated_altitude_part_function = interpolate.UnivariateSpline(line, interpolated_altitude[0:k+1],s=0,k=1)
            # 创建一个新的数组，从0到k+0.5，间隔为0.5
            # 这样整数位置是原数据，0.5部分为插值数据，做到在两个原数据间插值
            newline = np.linspace(0,k+0.5,2*(k+1))
            # a = np.linspace(0,1,2)
            #删掉0.5位置的值，因为此时0就是HeightTarget的插值值
            newline = np.delete( newline,[1])
            # 得到每隔一层插值一个的高度
            interpolated_part = interpolated_altitude_part_function(newline)
            # 将每隔一层插值一个的高度与后面的原数据拼接
            interpolated_altitude2 = np.append(interpolated_part, interpolated_altitude[k+1:len(interpolated_altitude)])

            # 计算插值后的温度等数据

            altitude_temperature = interpolate.UnivariateSpline(altitude, temperature,s=0,k=1)
            interpolated_temperature = altitude_temperature(interpolated_altitude2)
            altitude_water = interpolate.UnivariateSpline(altitude, water,s=0,k=1)
            interpolated_water= altitude_water(interpolated_altitude2)
            np.where(interpolated_water<0,0,interpolated_water)
            altitude_ozone = interpolate.UnivariateSpline(altitude, ozone,s=0,k=1)
            interpolated_ozone = altitude_ozone(interpolated_altitude2)
            np.where(interpolated_ozone<0,0,interpolated_ozone)
            altitude_pressure = interpolate.UnivariateSpline(altitude, pressure,s=0,k=1)
            interpolated_pressure = altitude_pressure(interpolated_altitude2)
            np.where(interpolated_pressure<0,0,interpolated_pressure)

            interpolated_altitude = interpolated_altitude2

            #加地面小于33层

        # if len(new_altitude) <34:

        # 画图专用 运行注释掉
        
        # plt.subplot(4,1,1)
        # plt.plot(altitude,temperature,'ro',interpolated_altitude,interpolated_temperature,'b*')
        # plt.title('H_T')
        
        # plt.subplot(4, 1, 2)
        # plt.plot(altitude,water,'ro',interpolated_altitude,interpolated_water,'b*')
        # plt.title('H_W')
        
        # plt.subplot(4, 1, 3)
        # plt.plot(altitude,ozone,'ro',interpolated_altitude,interpolated_ozone,'b*')
        # plt.title('H_O3')
        
        # plt.subplot(4, 1, 4)
        # plt.plot(altitude,pressure,'ro',interpolated_altitude,interpolated_pressure,'b*')
        # plt.title('H_P')
        # plt.show()

        # #插值结果
        atmos = {

            "altitude": interpolated_altitude,
            "pressure": interpolated_pressure,
            "temperature": interpolated_temperature,
            "water": interpolated_water,
            "ozone": interpolated_ozone
        }


    return atmos,RH
   
    # df.to_csv(r'C:\Users\1\Desktop\casestudy\1016-reverse.csv', index=False, sep=',')
def get_aero(aeropath,iPos,jPos):
        
        nc = Dataset(aeropath,'r')
        lon = 0.25 * jPos
        lat = 90 - 0.25 * iPos

        #0.25*0.25 to 1*1
        ib = int(lat+0.5)
        jb = int(lon+0.5)
        
        # Aero 的lon从-180，180，lat从-90，90，大小为180*360
        ib = 90+ib

        if jb <=180: 
            jb = jb+180

        if jb >180 : 
            jb = jb-180
        a = nc.variables['AOD550_mean'][:].data
        AOD550 = nc.variables['AOD550_mean'][:].data[ib,jb]
        AOD670 = nc.variables['AOD670_mean'][:].data[ib,jb]
        AOD870 = nc.variables['AOD870_mean'][:].data[ib,jb]
        AOD1600 = nc.variables['AOD1600_mean'][:].data[ib,jb]
        AOD = np.array([AOD550,AOD670,AOD870,AOD1600])

        return AOD
def get_aero_MODIS(aeropath,iPos,jPos):
            
            nc = Dataset(aeropath,'r')
            lat = 90 - 0.25 * iPos
            lon = 0.25 * jPos
            
    
            #0.25*0.25 to 1*1
            ib = int(lat+0.5)
            jb = int(lon+0.5)
            
            
            nc = Dataset(aeropath,'r')
            # AeroMODIS 的lon从-180，180，lat从-90，90，大小为180*360
            ib = 90+ib
    
            if jb <=180: 
                jb = jb+180
    
            if jb >180 : 
                jb = jb-180


            AOD550 = nc.variables['Aerosol_Optical_Thickness_550_Land_Mean'][:].data[ib,jb]
            AOD = np.array([AOD550])
            return AOD
def get_aero2(aeropath,iPos,jPos):
        
        Aero = np.fromfile(aeropath, dtype=np.float32)
        Aero.shape = 4,721, 1440

        # Aero 的lon从-180，180
        if jPos > 720:
            jb = jPos - 720

        if jPos <= 720:
            jb = jPos + 720

        #lat从-90，90
        ib = 721-iPos

        ib 
        AOD550 = Aero[0,ib,jb]
        AOD670 = Aero[1,ib,jb]
        AOD870 = Aero[2,ib,jb]
        AOD1600 = Aero[3,ib,jb]
        AOD = np.array([AOD550,AOD670,AOD870,AOD1600])

        return AOD


MatchTab = pd.read_csv(MatchTablePath,index_col = False)
Height = np.fromfile(HeightPath, dtype=np.float32)
Height.shape = 721, 1440
time0 = datetime(2021, 1, 1, 0, 0)

fp = open(sOutFile, 'w')
fp.write('lat,lon,ipos,jpos,B1o,B1s,Diff,B2o,B2s,Diff,B3o,B3s,Diff,B4o,B4s,Diff,B5o,B5s,Diff,B6o,B6s,Diff,B7o,B7s,Diff,B8o,B8s,Diff,B9o,B9s,Diff,B10o,B10s,Diff,B11o,B11s,Diff,B12o,B12s,Diff,B13o,B13s,Diff,B14o,B14s,Diff,B15o,B15s,Diff,B16o,B16s,Diff,B17o,B17s,Diff,B18o,B18s,Diff,B19o,B19s,Diff,NDVI,SZA,SAA,VZA,VAA,CLM,Time,Num,Height,AOD550,dTime\n')
nBand = 19
nMatch= len(MatchTab)
count = 0
fp.close()

# for i in np.arange(len(MatchTab)):
for i in np.arange(len(MatchTab)):
    
    i = i + Bias
    if MatchTab.loc[i, 'dTime'] >= dTime:continue
    X = SixS()

    

    fp = open(sOutFile, 'a+')

    if MatchTab.loc[i, 'NDVI'] <= 0:  continue
    # if MatchTab.loc[i, 'NDVI'] >= 0.6:  continue

    if MatchTab.loc  == 'S' :continue

    else:
        # time
        timeMERSI = datetime(MatchTab.loc[i, 'ERA5Year'], MatchTab.loc[i, 'ERA5Month'], MatchTab.loc[i, 'MERSI_Day'],
                         MatchTab.loc[i, 'MERSI_Hour'], MatchTab.loc[i, 'MERSI_Minute'], 0)
        timeMERSIstr = timeMERSI.strftime('%Y-%m-%d %H:%M:%S')
        timeVIIRS = datetime(MatchTab.loc[i, 'ERA5Year'], MatchTab.loc[i, 'ERA5Month'], MatchTab.loc[i, 'VIIRSDay'],
                         MatchTab.loc[i, 'VIIRSHour'], MatchTab.loc[i, 'VIIRSMinute'], 0)
        timeVIIRSstr = timeVIIRS.strftime('%Y-%m-%d %H:%M:%S')
        ERA5time = datetime(MatchTab.loc[i, 'ERA5Year'], MatchTab.loc[i, 'ERA5Month'], MatchTab.loc[i, 'ERA5Day'],
                            MatchTab.loc[i, 'ERA5Hour'], 0, 0)
        # Aerosol

        # ERA5
        # Aerofiles = glob.glob(Aeroncpath + '/*.nc')
        # Aerofile = []
        # AKeyword = '{0}{1}{2}'.format(str(ERA5time.year),str(ERA5time.month).zfill(2),str(ERA5time.day).zfill(2))
        # for k in Aerofiles:
        #     # 确定日期
        #     if AKeyword in os.path.basename(k):
        #         Aerofile = k
        #         break
        # Aero = get_aero(Aerofile,MatchTab.loc[i, 'iPos'], MatchTab.loc[i, 'jPos'])

        #MODISSample
        days = (ERA5time - time0).days + 1
        AeroMODIS_SampleFiles = glob.glob(AeroMODIS_Samplepath + '/*.dat')
        AeroMODIS_SampleFile = []
        CKeyword = 'MOD09CMA.A{}{}'.format(str(ERA5time.year), str(days).zfill(3))
        for k in AeroMODIS_SampleFiles:
            # 确定日期
            if CKeyword in os.path.basename(k):
                AeroMODIS_SampleFile = k
                break
        MODIS_Sample = np.fromfile(AeroMODIS_SampleFile, dtype=np.float32)
        MODIS_Sample.shape = 721, 1440
        Aero = np.array([MODIS_Sample[MatchTab.loc[i, 'iPos'], MatchTab.loc[i, 'jPos']]])

        
        # Aero2
        # Aerofiles2 = glob.glob(AeroSamplepath + '/*.dat')
        # Aerofile2 = []
        # AKeyword2 = '{0}{1}{2}'.format(str(ERA5time.year),str(ERA5time.month).zfill(2),str(ERA5time.day).zfill(2))
        # for k in Aerofiles2:
        #     # 确定日期
        #     if AKeyword2 in os.path.basename(k):
        #         # print('Find Match ' + time4Y2M2D)  # 绝对路径
        #         Aerofile2 = k
        #         break
        # Aero2= get_aero2(Aerofile2,MatchTab.loc[i, 'iPos'], MatchTab.loc[i, 'jPos'])

        # MODIS Aero

        # AeroMODISfiles = glob.glob(AeroMODISpath +'/*.nc')
        # AeroMODISfile = []
        # days = (ERA5time - time0).days + 1
        # AKeywordMODIS = 'A{}{}'.format(str(ERA5time.year), str(days).zfill(3))

        # for k in AeroMODISfiles:
        #     # 确定日期
        #     if AKeywordMODIS in os.path.basename(k):
        #         # print('Find Match ' + time4Y2M2D)  # 绝对路径
        #         AeroMODISfile = k
        #         break
        # k  = i
        # AeroMODIS = get_aero_MODIS(AeroMODISfile,MatchTab.loc[k, 'iPos'], MatchTab.loc[k, 'jPos']) 
        # Aero  = AeroMODIS


        ## Aerosol筛选条件
        if sum(Aero < 0.0001) > 0 : 
            continue
        # BRDF
        ## 差一天
        days = (ERA5time - time0).days + 1
        BKeyword = 'MCD43C1.A{}{}'.format(str(ERA5time.year), str(days).zfill(3))
        BRDFfiles = glob.glob(BRDFpath + '/*.dat')
        BRDFfile = []
        for k in BRDFfiles:
            # 确定日期
            if BKeyword in os.path.basename(k):
                # print('Find Match ' + time4Y2M2D)  # 绝对路径
                BRDFfile = k
                break
        BRDF = np.fromfile(BRDFfile, dtype=np.float32)
        BRDF.shape = 57, 721, 1440


        # 大气廓线
        HeightTarget = Height[MatchTab.loc[i, 'iPos'], MatchTab.loc[i, 'jPos']]
        
        atmos,RH = get_atmos(ERA5time, MatchTab.loc[i, 'iPos'], MatchTab.loc[i, 'jPos'],HeightTarget)
        if RH >= 90 :
            continue
        
        # save geo
        lon = 0.25 * MatchTab.loc[i, 'jPos']
        lat = 90 - 0.25 * MatchTab.loc[i, 'iPos']
        fp.write('{0},'.format('%.2f'%lat))
        fp.write('{0},'.format('%.2f'%lon))
        fp.write('{0},'.format(MatchTab.loc[i, 'iPos']))
        fp.write('{0},'.format(MatchTab.loc[i, 'jPos']))

        # geo&time
        VZA = MatchTab.loc[i, 'VZA_'+KEY]
        VAA = MatchTab.loc[i, 'VAA_'+KEY]
        # geo&time
        time = eval('time'+KEY+'str')
        X.geometry.from_time_and_location(lat, lon, time, VZA, VAA)

     
        # aerosol
        X.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        X.aot550 = Aero[0]
        # X.aero_profile = AeroProfile.UserProfile(AeroProfile.NoAerosols)
        # X.aero_profile.add_layer(Aero[0], 0.55)
        # X.aero_profile.add_layer(Aero[1], 0.67)
        # X.aero_profile.add_layer(Aero[2], 0.87)
        # X.aero_profile.add_layer(Aero[3], 1.60)
        
        X.atmos_profile = AtmosProfile.RadiosondeProfile(atmos)
        
        # 目标高度
        HeightTarget = Height[MatchTab.loc[i, 'iPos'], MatchTab.loc[i, 'jPos']]
        # X.altitudes.set_target_custom_altitude(atmos['altitude'].min())
        X.altitudes.set_target_custom_altitude(atmos['altitude'][0])
        # 传感器高度
        X.altitudes.set_sensor_satellite_level()
        

        
        for j in np.arange(nBand) :

            
            
            # 波长
            # Band = 'PredefinedWavelengths.ACCURATE_MODIS_AQUA_'+'{}'.format(j+1)
            # X.wavelength = Wavelength(int(Band))
            X.wavelength = Wavelength(eval(KEY)[j])

            BRDFBand = j
            # nBand = 1-19 for parameter 1 ...
            BRDFP1 = BRDF[BRDFBand ,MatchTab.loc[i,'iPos'],MatchTab.loc[i,'jPos']]
            BRDFP2 = BRDF[BRDFBand + 19, MatchTab.loc[i, 'iPos'], MatchTab.loc[i, 'jPos']]
            BRDFP3 = BRDF[BRDFBand + 38, MatchTab.loc[i, 'iPos'], MatchTab.loc[i, 'jPos']]
            BRDFj = np.array([BRDFP1,BRDFP2,BRDFP3])
            ## BRDF筛选条件
            if sum(BRDFj<= 0.00001) > 0 :
                    fp.write(f'{np.nan},')
                    fp.write(f'{np.nan},')
                    fp.write(f'{np.nan},')
                    continue



            X.ground_reflectance = GroundReflectance.HomogeneousMODISBRDF(BRDFj[0],BRDFj[1],BRDFj[2])
            X.run()
            res = X.outputs.apparent_radiance
            res2 = X.outputs.apparent_reflectance
            Obres = MatchTab.loc[i, f'B{j+1}_'+KEY]
            Diff = ((Obres-res)/res)*100
            fp.write('{0},'.format('%.5f'%Obres))
            fp.write('{0},'.format('%.5f'%res))
            fp.write('{0}%,'.format('%.5f'%Diff))

        fp.write('{0},'.format('%.5f'%MatchTab.loc[i,'NDVI']))
        fp.write('{0},'.format('%.5f'%MatchTab.loc[i,'SZA_'+KEY] ))
        fp.write('{0},'.format('%.5f'%MatchTab.loc[i,'SAA_'+KEY] ))
        fp.write('{0},'.format('%.5f'%MatchTab.loc[i,'VZA_'+KEY] ))
        fp.write('{0},'.format('%.5f'%MatchTab.loc[i,'VAA_'+KEY] ))
        fp.write('{0},'.format('%.5f'%MatchTab.loc[i,'CLM_'+KEY] ))
        fp.write('{0},'.format(time))
        fp.write('{0},'.format(i))
        fp.write('{0},'.format('%.5f'%HeightTarget))
        fp.write('{0},'.format('%.5f'%Aero[0]))
        fp.write('{0},'.format('%.1f'%MatchTab.loc[i, 'dTime']))
        fp.write('\n')
        fp.close()
    count += 1
    print( str(i)+'finish!' +str(count))

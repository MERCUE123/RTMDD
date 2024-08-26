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


FY3Dpath = r"D:\FY3D\1KM&GEO\FY3D_MERSI_GBAL_L1_20210302_1920_1000M_MS.HDF"

HDF = h5py.File(FY3Dpath, 'r')
E0 = h5py.AttributeManager(HDF).get('Solar_Irradiance')
E0 = E0 / 3.1415
RefSB1KM = np.array(HDF['Data/EV_1KM_RefSB'][:])
Bandn = RefSB1KM[4,:,:]
MAX = np.max(Bandn)
print(MAX)
RefSB250 = np.array(HDF['Data/EV_250_Aggr.1KM_RefSB'][:])
N, row, col = RefSB250.shape
cali = HDF['Calibration/VIS_Cal_Coeff'][:]  # 定标系数
# BandRadiance2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
BandRadiance = list(np.arange(0, 19))

k=1
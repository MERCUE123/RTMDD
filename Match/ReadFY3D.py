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

DATAPATH = r"D:\FY3D\1KM&GEO\FY3D_MERSI_GBAL_L1_20210329_1840_1000M_MS.HDF"
File = h5py.File(DATAPATH , 'r')
data = File['Data/EV_1KM_RefSB'][13]
plt.imshow(np.transpose(data))
plt.show()
x = data.max()
print(x)

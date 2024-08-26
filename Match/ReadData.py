# -*- coding: utf-8 -*-
import numpy as np
import h5py
import os
import  pandas as pd
import scipy.sparse as sp
from ctypes import *
from dateutil.parser import parse
from datetime import datetime
import matplotlib.pyplot as plt

landmaskfile =r'C:\Users\1\Desktop\casestudy\data\LandMask_0.25X0.25.dat'
Landmask = np.fromfile(landmaskfile, dtype=np.byte)
Landmask.shape = 721,1440

i = 368
j = 57
A = sum(sum(Landmask[i-1:i+2,j-1:j+2] > 0.5))
B = sum(Landmask[i-1:i+2,j-1:j+2] > 0.5)
print(A,B)
# x = fp

# fp.close()
# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import sys
from ctypes import *
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
import datetime
import pandas as pd
from scipy import interpolate
import os

NDVI_file = list([r'D:\MODIS\NDVI\MYD13C1.A2021057.061.2021076030617_NDVI_0.25X0.25.dat',
                 r'D:\MODIS\NDVI\MYD13C1.A2021073.061.2021090083039_NDVI_0.25X0.25.dat'])
a =1
import numpy as np
import h5py as h5py
import os
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sListFile = r"C:/Users/1/Desktop/casestudy/data/FY3D/1KM.dat"

fp = open(sListFile, 'r')
filelist = [x.rstrip() for x in fp]
fp.close()

sNightFile = sListFile.replace('1KM', '1K_Night')
# sDayFile = sListFile.replace('GEO1K', 'GEO1K_Day')

fp = open(sNightFile, 'w')
for k, sHdf in tqdm(enumerate(filelist)):
    if '.tmp' in filelist[k] :
        continue
    try:
            HDF = h5py.File(filelist[k], 'r')
            flag = str(HDF.attrs['Day Or Night Flag'])

            flag = flag[2]
            if flag =='N':
                fp.write(str(sHdf))
                fp.write('\n')
                HDF.close()

    except:FileNotFoundError
    continue

    # 0 代表白天

fp.close()

fp = open(sNightFile, 'r')
filelist = [x.rstrip() for x in fp]
fp.close()
for k, sHdf in tqdm(enumerate(filelist)):
    # kmfile = filelist[k].replace('GEO1K', '1000M')
    kmfile = filelist[k]
    CLMfile = kmfile.replace('FY3D_MERSI_GBAL_L1_','FY3D_MERSI_ORBT_L2_CLM_MLT_NUL_')
    # if os.path.exists(filelist[k]):
    #     os.remove(filelist[k])
    #     print(filelist[k])
    if os.path.exists(kmfile):
        os.remove(kmfile)
        print(kmfile)
    if os.path.exists(CLMfile):
        os.remove(CLMfile)
        print(CLMfile)




        



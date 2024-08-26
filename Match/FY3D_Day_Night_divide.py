import numpy as np
import h5py as h5py
import os
from tqdm import tqdm
import glob
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

Month = 'December'
GEOpath = r'D:/MERSI/'+Month+'/1KM&GEO&CLM'
def geo1k_1km_clm():
    sListFile = r"C:\Users\1\Desktop\casestudy\data\FY3D\GEO1KList.dat"
    sNightFile = r"C:\Users\1\Desktop\casestudy\data\FY3D\GEO1K_NightList.dat"
    list1 = open(sListFile, 'w')
    geofile = glob.glob(GEOpath +'/*GEO1K*.HDF')
    # geofile = glob.glob(GEOpath +'/*2021090[123456]*GEO1K*.HDF')
    list1.write('\n'.join(geofile))
    list1.close()
    sListFile = r"C:\Users\1\Desktop\casestudy\data\FY3D\GEO1KList.dat"
    
    fp = open(sListFile, 'r')
    filelist = [x.rstrip() for x in fp]
    fp.close()

    sNightFile = sListFile.replace('GEO1K', 'GEO1K_Night')
    sDayFile = sListFile.replace('GEO1K', 'GEO1K_Day')


    fp = open(sNightFile, 'w')
    for k, sHdf in tqdm(enumerate(filelist)):
        if '.tmp' in filelist[k] :
            continue
        try:
                HDF = h5py.File(filelist[k], 'r')
                flag = HDF['Timedata/DayNightFlag'][:]
                if np.sum(flag) >= 1:
                    fp.write(str(sHdf))
                    fp.write('\n')
                    HDF.close()

        except:FileNotFoundError
        continue

    fp = open(sNightFile, 'r')
    filelist = [x.rstrip() for x in fp]
    fp.close()

    for k, sHdf in tqdm(enumerate(filelist)):
        kmfile = filelist[k].replace('GEO1K', '1000M')
        CLMfile = kmfile.replace('FY3D_MERSI_GBAL_L1_','FY3D_MERSI_ORBT_L2_CLM_MLT_NUL_')
        if os.path.exists(filelist[k]):
            os.remove(filelist[k])
            print('remove'+filelist[k])
        if os.path.exists(kmfile):
            os.remove(kmfile)
            print('remove'+kmfile)
        if os.path.exists(CLMfile):
            os.remove(CLMfile)
            print('remove'+CLMfile)

        # 0 代表白天

    fp.close()
    
def onlyGEO1K ():
    
    sNightFile = r"C:\Users\1\Desktop\casestudy\data\FY3D\GEO1K_NightList.dat"
    fp = open(sNightFile, 'r')
    filelist = [x.rstrip() for x in fp]
    fp.close()
    filelistnew = []
    for k in range(len(filelist)):
        if len(filelist[k])>=100:
            continue
        else:
            filelist[k] = filelist[k].replace(r'C:\Users\1\Desktop\casestudy\data\FY3D\download', r'D:\MERSI\September\1KM&GEO&CLM')
            filelistnew.append(filelist[k])
    
    for k, sHdf in tqdm(enumerate(filelist)):
        kmfile = filelist[k].replace('GEO1K', '1000M')
        CLMfile = kmfile.replace('FY3D_MERSI_GBAL_L1_','FY3D_MERSI_ORBT_L2_CLM_MLT_NUL_')
        if os.path.exists(filelist[k]):
            os.remove(filelist[k])
            print('remove'+filelist[k])
        if os.path.exists(kmfile):
            os.remove(kmfile)
            print('remove'+kmfile)
        if os.path.exists(CLMfile):
            os.remove('remove'+CLMfile)
            print(CLMfile)


    return filelist


if __name__ =='__main__':
    geo1k_1km_clm()






   
   




        



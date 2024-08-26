import os
import sys
from ctypes import *
from datetime import datetime, timedelta
import numpy as np
import pyhdf.SD as HDF
import tqdm

sListFile = r'C:\Users\1\Desktop\py\Match\listMODIS_MYD03.dat'
deletedate = np.array([['2021-03-03'],['2021-03-04'],['2021-03-06'],['2021-03-07'],
                        ['2021-03-09'],['2021-03-11'],['2021-03-12'],['2021-03-14'],
                        ['2021-03-15'],['2021-03-17'],['2021-03-19'],['2021-03-20']
                        ,['2021-03-22'],['2021-03-23'],['2021-03-25'],['2021-03-26']
                        ,['2021-03-27'],['2021-03-28'],['2021-03-30']
                        ,['2021-03-31']])

def findmyd02(keyword, path):
    file = []
    files = os.listdir(path)
    for i in files:
        i = os.path.join(path, i)  # 合并路径与文件名
        # if os.path.isfile(i):#判断是否为文件
        # print(i)
        if keyword in os.path.basename(i):
            file = i
            break

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

    return file

if __name__ == '__main__':


    fp = open(sListFile, 'r')
    filelist = [x.rstrip() for x in fp]
    fp.close()
    time0 = datetime(2021, 1, 1, 0, 0)
    # sOutFile = sListFile.replace('DayNight_Delete', 'NightDelete')

    # fp = open(sOutFile, 'w')


    for k, sHdf in tqdm.tqdm(enumerate(filelist)):
        file = filelist[k]
        Keyword = os.path.basename(file)[5:19]

        #取出天数
        Timeday = Keyword[6:9]

        # 天数对应日期
        Time = time0 + timedelta(days=int(Timeday) - 1)

        # 转为str
        Timestr = Time.strftime('%Y-%m-%d')

        # 判断是否在删除日期内
        if Timestr in deletedate:
            

        # 找到对应的myd35和myd02
            MYD35path = findmyd35('MYD35_L2' + Keyword, os.path.dirname(file))
            MYD021KMpath = findmyd02('MYD021KM' + Keyword, os.path.dirname(file))
        # 删除
            if os.path.exists(filelist[k]):
                os.remove(filelist[k])

            if MYD35path != [] :
                     os.remove(MYD35path)
            else:
                continue

            if MYD021KMpath != [] :
                    os.remove(MYD021KMpath)
            else:
                continue

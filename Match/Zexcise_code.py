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
import functools
import math
import re
## 对数字的余数进行排序
def Code1():
    numbers = [1, 3, 4, 2, 5]
    sorted_numbers = sorted(numbers,key=lambda x: x%3,reverse=True)
    return sorted_numbers

## 对字母进行排序
def Code2():
    letters = ['aa', 'kb', 'jc', 'ed', 'de']
    ## 按第二个字母排序
    sorted_letters = sorted(letters,key=lambda x: x[1])
    return sorted_letters

## 连续去重字母排序
## set()去重
def Code2p1():
    letters = ['aafasfvbhrehtt','asgwgasjghr']
    letters = list(letters[0]+letters[1])
    unique_letters = list(set(letters))
    ## 按第二个字母排序
    sorted_letters = sorted(unique_letters,key=lambda x: x[0])
    return sorted_letters

##字典去重排序
def Code2p2():
    letters = 'Hello World!'
    chardict = {}
    # letters = list(letters[0])
    for char in letters:
        chardict[char] = chardict.get(char,0)+1
    unique_letters = chardict.keys()
    ## 按第二个字母排序
    sorted_letters = sorted(unique_letters,key=lambda x: x[1])
    return sorted_letters


## 数字加字字母排序
def Code3():
    numbers_letters = ['1a', '3b', '4c', '2d', '5e']
    ## 按先数字后字母排序
    sorted_numbers_letters = sorted(numbers_letters,key=lambda x: (int(x[0]),x[1]))
    return sorted_numbers_letters

## zip函数
def Code4():
    ## 通过zip快速创建字典
    scores = [1, 3, 4, 2, 5]
    students = ['a', 'b', 'c', 'd', 'e']
    studict = dict(zip(students,scores))
    return studict

## map函数
def Code5():
    ## 通过map函数对列表中的元素进行操作
    numbers = [11,45,14,11,45,14]
    # List是存储单列数据的集合，Map是map类型存储键值对这样的双列数据的集合；
    # mapn = map(lambda x: x*x,numbers)
    squared_numbers = list(map(lambda x: x*x,numbers))
    return squared_numbers

## filter函数
def Code6():
    ## 通过filter函数对列表中的元素进行筛选
    numbers = [11,45,14,11,45,14]
    filtered_numbers = list(filter(lambda x: x>20,numbers))
    filtered_numbers2 = list(filter(lambda x: x%2!=0,numbers))
    return filtered_numbers2

## reduce函数
def Code7():
    numbers = Code6()
    # reduce需要两个参数，lambda x: x**2 不行
    res = functools.reduce(lambda x,y: x*y,Code6())
    return res 

def Code8():

## split函数
    ## 通过split函数对字符串进行分割
    string = 'hello,world,python'
    split_string = string.split(',')
    ## .upper小转大,.lower大转小
    split_string = list(map(lambda x: x.upper(),split_string))
    return split_string

## join函数
def Code9():
    ## 通过join函数对字符串进行合并
    split_string = Code8()
    ## join前面每个字母都加一个空格
    join_string = ' '.join(split_string)
    return join_string

## os获取文件名
def Code10():
    file = __file__
    abspath = os.path.abspath(file)
    basename = os.path.basename(file)
    dirname = os.path.dirname(file)
    return '_'.join([basename,dirname])

## strip函数
def Code11():
    ## 通过strip函数对字符串进行去除两端
    string = '####hello,                       world,p   yt ho   n  ####'
    strip_string = string.strip("#")
    # strip_string2 = filter(' ',strip_string)
    return strip_string

## replace函数
def Code12():
    ## 通过replace函数对字符串进行替换
    string = 'hello,world,python'
    replace_string = string.replace(',','   ')
    return replace_string

## enumerarte函数 e nume rate
def Code13():
    for i , letter in enumerate(Code2()):
        print(i,letter)

## range函数
def Code14():
    numbers1 = range(5)
    numbers2 = range(1,5)
    ## start,stop,step
    numbers3 = list(range(1,5,2))

    for i in range(len(Code1())):
        print(i,Code2()[i])
    return numbers1,numbers2,numbers3

## all -- any
def Code15():
    numbers = [1, 1,4,5,1,4]
    judgeall = all([x>=1 for x in numbers])
    judgeany = any([x>=5 for x in numbers])
    return judgeall,judgeany

## eval函数
## eval()函数就是实现list、dict、tuple、与str之间的转化
def Code16():
    ## 1.让字符串指定为变量
    str1 = 'numbers1'
    numbers1 = [1,1,4,5,1,4]
    numbers2 = [11,45,14]
    eval_str2var = eval(str1)
    ## list、dict、tuple、与str之间的转化
    str2 = '[[1,1],[4,5],[1,4]]'
    eval_str2list = eval(str2)
    return eval_str2var,eval_str2list,type(eval_str2list)

## open文件方法
def Code17():
    file = '*.txt'
    with open(file,'r') as f:
        lines = f.readlines()
    return 0

## reversed函数
def Code18():
    numbers = [1,1,4,5,1,4]
    reversed_numbers = list(reversed(numbers))
    for k in range(int((len(numbers))/2)):
        temp = numbers[k]
        numbers[k] = numbers[len(numbers)-1-k]
        numbers[len(numbers)-1-k] = temp
    ## 反转再反转
    reversed_numbers2 = numbers[::-1]
    return numbers,reversed_numbers,reversed_numbers2

## axis&drop函数
def Code19():
    array2 = np.arange(12).reshape(3,4)
    ## 从对应的轴去求和，相当于串串
    # npaxis0 = np.sum(array2,axis=0)
    # npaxis1 = np.sum(array2,axis=1)
    # npaxis2 = np.sum(array2,axis=2)
    ## pandas主要处理二维数据
    array3 = pd.DataFrame(array2.reshape(3,4))
    # pdaxis0 = array3.sum(axis=0)
    # pdaxis1 = array3.sum(axis=1)
    ## drop
    return 0

## pandas筛选数据
def Code20():
    array = pd.DataFrame({'A':[1,2,3,4,5],'B':[6,7,8,9,10],'C':[11,12,13,14,15]})

    ## 通过列名筛选数据
    array1 = array[array['A']>=3]
    ## 通过行索引筛选数据
    array2 = array.drop(array[array['A']>=3].index,axis=0)
    A = np.pi
    ## fstring的位数保留
    print(f'pi的值是{A:.4f}')
    return array1,array2

## type()、len()等优先使用
def Code21():
    # .len会报错，但len()不会
    str1 = 'asfasfwf'
    return len(str1),type(str1)

## lambda函数综合
def Code22():
    ## 字母排序
    friut = ['Apple','Banana','Orange','Peach','Grape']
    pairs = list([(1, 'one'), (3, 'two'), (2, 'three'), (4, 'four')])
    numbers = list([1, 1, 4, 5, 1, 4])
    sorted_friut = sorted(friut,key=lambda x: x[0].upper())
    ## 长度排序
    sorted_friut = sorted(friut,key=lambda x: len(x))
    ## 混合排序
    sorted_pairs = sorted(pairs,key=lambda pairs : len(pairs[1]))
    ## 筛选 list(filter) filter不能直接输出,map同理
    selected = list(filter(lambda pairs : len(pairs[1])>=4,pairs))
    ## 数字计算 结合map
    numbers = list(map(lambda x : x**2,numbers))
    ## 累加或累乘
    addall = functools.reduce(lambda x,y:x+y,numbers)

    return selected,numbers,addall

## 数字处理相关
def Code23():
    pi = np.pi
    ## 四舍五入
    roundpi = round(np.pi)
    print(f'pi四舍五入后为{roundpi}')
    ## 向下取整
    floorpi = math.floor(pi)
    print(f'floorpi为{floorpi}')
    intpi = int(pi)
    print(f'intpi为{intpi}')
    ## 向上取整
    ceilpi = math.ceil(pi)
    print(f'ceilpi为{ceilpi}')
    ## 取余数
    remainpi = pi%3
    print(f'pi除以3的余数remainpi为{remainpi}')
    ## 绝对值
    absnum = abs(1+2j)
    print(f'1+2j的绝对值为为{absnum}')
    return 0

## try except
def Code24():
    try:
        # 可能引发多种异常的代码
        filename = input("请输入文件名：")
        with open(filename, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("文件未找到。")
    except PermissionError:
        print("没有权限读取文件。")
    except Exception as e:
        # 捕获其他所有类型的异常
        print(f"发生了一个未知错误: {e}")
    return 0


## 正则表达式 --- 结合Unix表达式
def Code25():

    demo = 'My email is example@example.com. You can reach me there.' \
           'Visit my blog at https://www.example-blog.com for more info.' \
           'Our meeting is scheduled for 2023-06-22. Please mark your calendar'


    ## +表示前面的出现n次，?表示0或1个 *表示0或n次 \d{3}匹配3个数字  \d{1，4}匹配1-4个数字 \b匹配独立单词 如use 而非beused
    ## \d一个数字 \d+多个数字  *? []?  +? {}？尽可能少的匹配前面的参数
    demo1 = re.findall(r'\d',demo)
    demo1_1 = re.findall(r'\d+',demo)
    demo1_2 = re.findall(r'\d?',demo)
    demo1_3 = re.findall(r'\d{4}', demo)
    ## \D 不是数字
    demo2 = re.findall(r'\D+',demo)
    ## []找括号里的东西
    demo3 = re.findall('\\be[a-z]+\\b',demo)
    print (demo3)


## isinstance查找类型
def Code26():
    my_list = [1,1,4,5,1,4]
    my_tuple = (1,2,3,4,5,6)
    if isinstance(my_list,tuple):
        print('this is list')
    else:
        print(('this is'+str(type(my_list))))
    if isinstance(my_tuple,tuple):
        print('this is tuple')
    # print(type(my_tuple))

## os相关内容
def Code27():
    ## 判断是否为文件
    print(os.path.isfile(__file__))
    ## 判断是否为目录
    print(os.path.isdir(os.path.dirname(__file__)))
    ## 判断是否存在
    print(os.path.exists('a.txt'))
    ##os.mkdir()创建目录
    ##os.rmdir()删除目录
    ##os.remove()删除文件
    print(os.stat(__file__ ))
    ## os.rename(new_name,old_name)

## slice切片
def Code28():
    ## slice可以用来切片，range不行 range可以创建数组，slice不行
    numbers = [1,1,4,5,1,4,1,1,5,1,4]
    number_slice =numbers[slice(0,10,2)]
    number_slice2 = numbers[0:10:2]
    # 列表表达式
    number_range = [i for i in range(0,10,2)]
    print(number_slice,number_slice2,number_range)

## count 和 index 和 find
def Code29():
    numbers = [1,1,4,5,1,4,1,1,4,5,1,4]
    string = 'I love 1 and 2 and 3'
    numbers_2d = np.array(numbers).reshape(3,4)
    ## 计数
    bool_numbers = all([x <= 1 for x in numbers])
    count_numbers = numbers.count(1)
    number_index = numbers.index(1)
    string_find = string.find('1')
    print(count_numbers,bool_numbers,number_index,string_find)

## 列表推导式
def Code30():
    numbers = [1,1,4,5,1,4,1,1,4,5,1,4]
    numbers2  = [1,1,4,5,1,4,1,1,4,5,1,4]
    ##建立列表
    set_list1 = [x for x in range(0,10,2)]
    ## 筛选
    filter_list = [x for x in numbers if x>=4]
    filter_list2 = list(filter(lambda x : x>=4,numbers))
    ## 合并
    merge_list = [(x,y) for x in numbers for y in numbers2]
    zip_list = list(zip(numbers,numbers2))
    print(set_list1,filter_list,filter_list2,merge_list,'\n',zip_list)


if __name__ =='__main__':
    Code30()
    ## pass起到占位 啥也不干
    ## contine 跳过 当前 循环
    ## break 跳出 整个 循环

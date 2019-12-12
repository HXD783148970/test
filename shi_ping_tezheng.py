# *-* coding:utf8 *-*
'''
文件夹下的txt震动信号进行时域和频域特征提取
'''
import pandas as pd
import numpy as np
import math
import os
from scipy import integrate
import glob
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import KFold,cross_val_score as cvs,train_test_split as TTS
from time import time
import datetime
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from scipy.fftpack import fft, fftshift, ifft
from scipy import fftpack
import math

csv_list = glob.glob('C:/Users/1701/Desktop/1/*.txt') 
print(u'共发现%s个txt文件'% len(csv_list))
print(u'正在处理............')
a_pingyuzhi = []
a_shiyuzhi = []
a_baoluo = []
v_pingyuzhi =  []
v_shiyuzhi = []
v_baoluo = []

for i in csv_list: #循环读取同文件夹下的csv文件
    #读TXT文件
    with open('file.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
    with open(i, 'rb') as filein:
        for line in filein:
            line_list = line.decode().strip().split(",")
            A=pd.DataFrame(line_list)
    data = []
    for line in open(i,"r"): #设置文件对象并读取每一行文件
        data.append(line)       
#读采样频率
    totalCount =data[1]
    pinglv = re.sub("\D", "", totalCount) 
#计算要取得数据长度
    long_data = 2*int(pinglv)
#取数据
    new_data = A.iloc[1:long_data+1,0]
#求指标
    records1 = np.array(new_data)
    records1 = records1.astype(float)

    #时域指标
    def  psfeatureTime(data):
    #均值
        df_mean=data.mean()
        df_var=data.var()
        df_std=data.std()
    #均方根
        df_rms=np.sqrt(pow(df_mean,2) + pow(df_std,2))
    #峰峰值
        fengfengzhi = max(data)-min(data)
    #偏度
        df_skew=pd.Series(data).skew()
    #峭度
        df_kurt=pd.Series(data).kurt()
        sum=0
        for i in range(len(data)):
            sum+=np.sqrt(abs(data[i]))
    #波形因子
        df_boxing=df_rms / (abs(data).mean())
    #峰值因子
        df_fengzhi=(max(data)) / df_rms
    #脉冲因子
        df_maichong=(max(data)) / (abs(data).mean())
    #裕度因子
        df_yudu=max(data)/ pow(sum/(len(data)),2)
    #峭度
        df_qiaodu  =(np.sum([x**4 for x in data])/len(data)) / pow(df_rms,4)
        featuretime_list = [round(df_rms,3),round(fengfengzhi,3),round(df_fengzhi,3),round(df_boxing,3),round(df_maichong,3),round(df_yudu,3),round(df_qiaodu,3)]
        return  featuretime_list
    p3 =psfeatureTime(records1)
    a_shiyuzhi.append(p3)
    #傅里叶变换
    a_fft = fft(records1) / 10000
    #频域指标
    def pingyu(df):
        f1 = np.sum([x for x in df])/len(df)
        f2 = np.sum([(x-f1)**2 for x in df]) / (len(df)-1)
        f3 = np.sum([(x-f1)**3 for x in df])/ (((math.sqrt(f2))**3)*len(df))
        f4 = np.sum([pow((x-f1),4) for x in df]) /(pow(f2,2)*len(df))
        p_y = [round(f1,3),round(f2,3),round(f3,3),round(f4,3)]
        return p_y
    a_pingyu =pingyu(a_fft)
    a_pingyuzhi.append(a_pingyu)
    #hilbert变换
    a_hil = fftpack.hilbert(records1)
    a_hil_fft = fft(abs(a_hil)) / 10000
    a_hil_pingyu =pingyu(a_hil_fft)
    a_baoluo.append(a_hil_pingyu)
    #求速度
    s=[]
    n=0
    for i in range(1,len(records1)):
        q= (records1[i-1],records1[i])
        s.append(integrate.simps(q))
    p_v = np.array(s)
    p_v = p_v.astype(float)
    p_v =psfeatureTime(p_v)
    v_shiyuzhi.append(p_v)
    #fft求频域指标
    v_fft = fft(s) / 10000
    v_pingyu =pingyu(v_fft)
    v_pingyuzhi.append(v_pingyu)
    #hiber变换
    v_hil = fftpack.hilbert(s)
    v_hil_fft = fft(abs(v_hil)) / 10000
    v_hil_pingyu =pingyu(v_hil_fft)
    v_baoluo.append(v_hil_pingyu)
    data.clear()
    s.clear()
    

a_q = pd.DataFrame(a_shiyuzhi)
a_fft_q = pd.DataFrame(a_pingyuzhi)
a_hil_q = pd.DataFrame(a_baoluo)

v_q = pd.DataFrame(v_shiyuzhi)
v_fft_q = pd.DataFrame(v_pingyuzhi)
v_hil_q = pd.DataFrame(v_baoluo)


colums_label=['a有效值','a峰峰值','a峰值','a波形','a脉冲','a裕度','a峭度'
              ,'af1','af2','af3','af4','a包络f1','a包络f2','a包络f3','a包络f4'
              ,'v有效值','v峰峰值','v峰值','v波形','v脉冲','v裕度','v峭度'
              ,'vf1','vf2','vf3','vf4','v包络f1','v包络f2','v包络f3','v包络f4']
        
list= [a_q,a_fft_q,a_hil_q,v_q,v_fft_q,v_hil_q]
f_data=pd.concat(list,axis=1)
f_data.columns = colums_label

print('..............处理完成...................')

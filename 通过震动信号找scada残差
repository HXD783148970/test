# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:16:02 2019
通过震动信号找到对应的scada残差数据
@author: 1701
"""
import pandas as pd
import numpy as np
import datetime
import os
import re
import time
import math
from scipy import integrate
import glob
import csv
import datetime
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
from scipy.fftpack import fft, fftshift, ifft
from scipy import fftpack

#读残差
data = pd.read_csv('C:\\Users\\1701\Desktop\\3895cancha.csv',index_col = False)
#读震动信号
file_path = "C:/Users/1701/Desktop/84"
path_list = os.listdir(file_path) # os.listdir(file)会历遍文件夹内的文件并返回一个列表
path_name=[]
# 利用循环历遍path_list列表并且利用split去掉后缀名
for i in path_list:
    path_name.append(i.split(".")[0])
# 排序一下
path_name.sort()

for file_name in path_name:
    # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    with open("save.txt","a") as f:
        f.write(file_name + "\n")
        # print(file_name)
    f.close()
#去除字符串中的空格
b = pd.Series([path_name[i].replace(' ', '')  for i in range(len(path_name)) ])
#风机编号为2位数时使用
q = []
for  i in b:
    s = i
    date = datetime.datetime(year=int(s[17:21]), month=int(s[21:23]),  day=int(s[23:25]),hour=int(s[26:28]),minute=int(s[28:30]),second=int(s[30:32]))
    aa = date.strftime("%Y-%m-%d %H:%M:%S")
    q.append(aa)
#震动信号的时间chuo
w = []  
for i in q:
    timestr = i
    timestamp = int(time.mktime(time.strptime(timestr,"%Y-%m-%d %H:%M:%S")))
    w.append(timestamp)

w_scada = data.loc[:,'timestape_max']

scada  = [] # 对应震动信号的scada数据
zd = [] # 震动信号
for i in w:#读出scada的时间点
    down = int(i/600)*600
    up = (int(i/600)*600) + 600
    for j in data.loc[:,'timestape_max']:
        if j>= down and j <= up:
            scada.append(data[data.loc[:,'timestape_max'] == j])
            zd.append(int(i))
y1 = pd.concat(scada,axis=0)       #符合条件的scada数据     


#通过时间戳找符合条件的震动信号
y = []

# # 输入秒级的时间，转出正常格式的时间
def timeStamp(timeNum):
    timeStamp = float(timeNum)
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y/%m/%d %H:%M:%S", timeArray)
    return otherStyleTime
for i in zd:
    aa = timeStamp(i)
    a = re.sub("\D", "", aa) #转为字符串
    b = list(a)
    b.insert(8,"-")#只能跑一次
    b  = 'C:/Users/1701/Desktop/84\\1076701-84-501-6_' + ''.join(b) + '.txt'  #84和21需要改
    y.append(b)
    
#取出震动信号数据
csv_list = glob.glob('C:/Users/1701/Desktop/84/*.txt') 
print(u'共发现%s个txt文件'% len(csv_list))

a_pingyuzhi = []
a_shiyuzhi = []
a_baoluo = []
v_pingyuzhi =  []
v_shiyuzhi = []
v_baoluo = []

for i in csv_list: #循环读取同文件夹下的csv文件
    if i in y:
    #读TXT文件
        print(u'正在处理....'+str(i))
        with open('file.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
        with open(i, 'rb') as filein:
            for line in filein:
                line_list = line.decode().strip().split(",")
                A=pd.DataFrame(line_list)
        data = []
        for line in open(i,"r"): #设置文件对象并读取每一行文件
            data.append(line)       
    ##读采样频率
    #    totalCount =data[1]
    #    pinglv = re.sub("\D", "", totalCount) 
    #计算要取得数据长度
     #   long_data = int(pinglv)
    #取数据
        long_data = 12800
        new_data = A.iloc[1:long_data,0]
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
        a_fft = abs(a_fft)
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
        a_hil_fft = abs(a_hil_fft)
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
        v_fft = abs(v_fft)
        v_pingyu =pingyu(v_fft)
        v_pingyuzhi.append(v_pingyu)
        #hiber变换
        v_hil = fftpack.hilbert(s)
        v_hil_fft = fft(abs(v_hil)) / 10000
        v_hil_fft = abs(v_hil_fft)
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


colums_label=['a_youxiao','a_fengfengzhi','a_fengzhi','a_boxing','a_maichong','a_yudu','a_qiaodu'
              ,'af1','af2','af3','af4','a_baoluof1','a_baoluof2','a_baoluof3','a_baoluof4'
              ,'v_youxiao','v_fengfengzhi','v_fengzhi','v_boxing','v_maichong','v_yudu','v_qiaodu'
              ,'vf1','vf2','vf3','vf4','v_baoluof1','v_baoluof2','v_baoluof3','v_baoluof4']
        
list= [a_q,a_fft_q,a_hil_q,v_q,v_fft_q,v_hil_q]
f_data=pd.concat(list,axis=1)
f_data.columns = colums_label

f_data['cancha']=  y1.loc[:,'cancha'].tolist()
f_data.to_csv('C:/Users/1701/Desktop/shi_ping.csv',index=False)

print('..............恭喜您，处理完成...................')
    
    


















#归一化
#x_nor = (data - data.min(axis=0))/ (data.max(axis=0)-data.min(axis=0))
#x_nor.to_csv('C:\\Users\\1701\\Desktop\\wt03.csv',index=False)
#标准化
#a=StandardScaler().fit_transform(data)
#scaler =(data-data.mean(axis=0))/data.std(axis=0)
#scaler.to_csv('C:\\Users\\1701\\Desktop\\wt133333.csv',index=False)

#复合滤波
#f = []
#for i in range(0,32):
#    col = data.iloc[:,i].values
#   
#    im4=list(col)
##中值滤波
#    data_F=signal.medfilt(im4,7)# 只能以奇数进行赋值
#    f.append(data_F)
###进行σ = 2的高斯滤波
##im2 = filters.gaussian_filter(im4,3)
#
##plt.scatter(data.Wspd_avg,data_F,s=4)
#pd.DataFrame(f).T.to_csv('C:\\Users\\1701\\Desktop\\wt2.csv',index=False)
#a=pd.DataFrame(f).T

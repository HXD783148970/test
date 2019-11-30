# import winpython
#coding=utf-8
from pyhht.emd import EMD
import pyhht.emd
# from pyhht.eemd import eemd
import numpy as np
from pyhht.visualization import plot_imfs
import matplotlib.pyplot as plt
# x=[i for i in range(100)]
# x=np.array(x)
import numpy as np
t = np.linspace(0, 1, 1000)
modes = np.sin(2 * 3.14 * 5 * t) + np.sin(2 * 3.14 * 10 * t)+np.sin(2 * 3.14 * 15 * t)+np.sin(2 * 3.14 * 40 * t)+np.sin(2 * 3.14 * 115 * t)
x=modes
imfsum=np.zeros((6,1000))
imfseemd1=np.zeros((1,1000))
#imfs3=np.zeros((1,1000))
#imfs4=np.zeros((1,1000))
decomposerEmd = EMD(x, n_imfs=6)
imfsemd=decomposerEmd.decompose()

for i in range(5):
    noise=np.random.normal(0,1,1000)#标准正态分布，均值0和方差1,1000个点
    y = x +noise
    decomposer = EMD(y,n_imfs=5)
    imfs = decomposer.decompose()
    imfsum=imfs+imfsum
imfseemd=imfsum/6

imfseemd=imfseemd.reshape((6,1000))


plot_imfs(y,imfseemd,t)#emd分解

plot_imfs(x,imfsemd,t)#emd分解


#print(imfs1.shape)
# plt.plot(modes)
#plt.plot(t,imfs1)
#plt.plot(t,imfs2)
#plt.plot(t,imfs3)
#plt.plot(t,imfs4)


# print(imfs1)
# print(imfs.shape)
# decomposer=EMD(x，n_imfs=5)
# imfs = decomposer.decompose()
#绘制分解图
# plot_imfs(x,imfs)
# help(EMD)
# x=np.random.normal(0,1,1000)
# plt.plot(x)
# help(np.array)
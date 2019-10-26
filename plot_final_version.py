# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 10:38:16 2017

@author: user
"""
import numpy as np
import matplotlib.colors
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

from matplotlib.colors import LogNorm

#import linecache
#
#print(linecache.getline("E:\\88-滴滴数据new\\6-路网识别\\result_angle_x_1000.txt",111))

if __name__ == "__main__":

    #Input the number of square that need to be plotted.
    Read = 111
    
    #一公里网格  0.011741653/2 0.0089946279/2  33*33
    Xnum = 33
    Ynum = 33
    Xstep = 0.011741653
    Ystep = 0.0089946279
    X0 = 116.18995243500 - Xstep/2
    Y0 = 39.75679276210 - Ystep/2
    file_out_angle_x = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_x_1000.txt", 'r')
    file_out_angle_y = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_y_1000.txt", 'r')
    file_out_num = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_num_1000.txt", 'r')
    
#    #500米网格  0.005870826/2 0.0044973139/2  67*66
#    Xnum = 67
#    Ynum = 66
#    Xstep = 0.005870826
#    Ystep = 0.0044973139
#    X0 = 116.18701702200 - Xstep/2
#    Y0 = 39.75454410520 - Ystep/2
#    file_out_angle_x = open("E:\\88-滴滴数据new\\6-路网识别\\result_angle_x_500.txt", 'r', encoding='utf-8')
#    file_out_angle_y = open("E:\\88-滴滴数据new\\6-路网识别\\result_angle_y_500.txt", 'r', encoding='utf-8')
#    file_out_num = open("E:\\88-滴滴数据new\\6-路网识别\\result_num_500.txt",'w')
    
#    #200米网格  0.00234833/2 0.0017989256/2  166*165
#    Xnum = 166
#    Ynum = 165
#    Xstep = 0.00234833
#    Ystep = 0.0017989256
#    X0 = 116.18525577400 - Xstep/2
#    Y0 = 39.75319491100 - Ystep/2
#    file_out_angle_x = open("E:\\88-滴滴数据new\\6-路网识别\\result_angle_x_200.txt", 'r', encoding='utf-8')
#    file_out_angle_y = open("E:\\88-滴滴数据new\\6-路网识别\\result_angle_y_200.txt", 'r', encoding='utf-8')
#    file_out_num = open("E:\\88-滴滴数据new\\6-路网识别\\result_num_200.txt", 'r', encoding='utf-8')

    data = np.empty(shape=[0, 5])
    
    cnt_vx = 0
    a_file_vx = file_out_angle_x.readlines()
    the_file_vx = a_file_vx[Read]
    spt_vx = the_file_vx.split(",")
    del(spt_vx[0])
    for i in spt_vx:
        cnt_vx += 1
    
    cnt_vy = 0
    a_file_vy = file_out_angle_y.readlines()
    the_file_vy = a_file_vy[Read]
    spt_vy = the_file_vy.split(",")
    del(spt_vy[0])
    for i in spt_vy:
        cnt_vy += 1

    cnt = 0
    for numx in spt_vx:
        numy = spt_vy[cnt]
        if (float(numx)*float(numx) <= 100*100) and (float(numy)*float(numy) <= 100*100):
            data = np.append(data, [[float(numx), 1,0,0,0]], axis=0)
            data[cnt][1] = float(numy)
            cnt += 1

    cnt = 0
    for num in spt_vx:
        R = math.sqrt(data[cnt,0]*data[cnt,0] + data[cnt,1]*data[cnt,1])
        data[cnt,2] = data[cnt,0]/R
        data[cnt,3] = data[cnt,1]/R
        temp = math.acos(data[cnt,0]/R)
        if data[cnt,1] < 0:
            temp = temp + math.pi
        data[cnt][4] = temp
        cnt += 1
    
#    f(theta)
#    delta_theta = 2*math.pi/100
#    for i in range(0,cnt):
#        temp = math.floor(data[i][4]/delta_theta)
#        f(temp) = f(temp)+1
        
    plt.plot( data[range(1,cnt),2], data[range(1,cnt),3], 'b.', markersize=1)
    plt.figure()
    plt.hist(data[range(1,cnt),4],30)
#    plt.figure()
#    plt.hist(data[range(1,cnt),5],30)
    
    plt.figure()
    plt.plot( data[range(1,cnt),0], data[range(1,cnt),1], 'b.', markersize=1)
    
    # normal distribution center at x=0 and y=5
#    x = np.random.randn(100000)
#    y = np.random.randn(100000) + 5

    plt.figure() 
    plt.hist2d(data[range(1,cnt),0], data[range(1,cnt),1], bins=40, norm=LogNorm())
    plt.colorbar()
    plt.show()

    
    
    #关闭文件
    file_out_angle_x.close()
    file_out_angle_y.close()
    file_out_num.close()
    
    file_out_data = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_data_1000.txt", 'w')
    for i in range(1,cnt):
        file_out_data.write(str(i))
        for j in range(0,5):
            file_out_data.write(','+str(data[i,j]))
        file_out_data.write("\n")
    file_out_data.close()
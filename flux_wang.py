# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:46:51 2017

@author: Liang Wang
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
import codecs
import os
import matplotlib.path as ppath

def exchange(a, b):
    return b,a

#参考算法：
#http://blog.csdn.net/maliang_1993/article/details/51517602
#矩阵行列式,行列式小于0，说明p1(x1,y1)在p2(x2,y2)顺时针方向上
def det(x1,y1,x2,y2):
    return x1*y2-x2*y1

#当同时满足：
#（1）CA和CB在CD的两侧（即一个顺时针方向上，一个在逆时针方向上）
#（2）AC和AD在AB的两侧（即一个顺时针方向上，一个在逆时针方向上）
#A(A1,B1),B(A2,B2),C(x1,y1),D(x2,y2)
#时AB可肯定和CD相交。 
def segment_point(x1,y1,x2,y2,A1,B1,A2,B2):
    #CA和CB在CD的两侧（即一个顺时针方向上，一个在逆时针方向上）
    #CD(x2-x1,y2-y1),CA(A1-x1,B1-y1),CB(A2-x1,B2-y1)
    #   det(x2-x1,y2-y1,A1-x1,B1-y1)*det(x2-x1,y2-y1,A2-x1,B2-y1) < 0
    det1 = det(x2-x1,y2-y1,A1-x1,B1-y1)*det(x2-x1,y2-y1,A2-x1,B2-y1)
    #AC和AD在AB的两侧（即一个顺时针方向上，一个在逆时针方向上）
    #AB(A2-A1,B2-B1),AC(x1-A1,y1-B1),AD(x2-A1,y2-B1)
    #   det(A2-A1,B2-B1,x1-A1,y1-B1)*det(A2-A1,B2-B1,x2-A1,y2-B1) < 0
    det2 = det(A2-A1,B2-B1,x1-A1,y1-B1)*det(A2-A1,B2-B1,x2-A1,y2-B1)
    #horizon
    if det1<0 and det2<0 and B1>B2 and y1==y2:  #向南
        return 1
    if det1<0 and det2<0 and B1<B2 and y1==y2:  #向北
        return -1
    #vertical
    if det1<0 and det2<0 and A1>A2 and x1==x2:  #向西
        return 2
    if det1<0 and det2<0 and A1<A2 and x1==x2:  #向东
        return -2
    else:
        return 0

##测试
#segment_point(1,3,5,3,2,5,1.5,1)
#segment_point(1,3,5,3,1.5,1,2,5)
#segment_point(1,3,5,3,-2,5,-3,1)
#segment_point(1,3,1,5,0,4,2,3)
#segment_point(1,3,1,5,2,3,0,4)
#segment_point(1,3,1,5,-3,4,0,3)

if __name__ == "__main__":

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    file_out = open("E:\\88-滴滴数据new\\1-data\\result_horizontal.txt",'w')
    file_out1 = open("E:\\88-滴滴数据new\\1-data\\result_vertical.txt",'w')
    path_dir = "E:\\88-滴滴数据new\\1-data\\test2\\"
    all_dir = os.listdir(path_dir)
    cnt = 0
    for a_file in all_dir:
        cnt += 1
        print("processing batch "+str(cnt)+"...")
        path = "E:\\88-滴滴数据new\\1-data\\test2\\" + a_file
        print(path)
        file = open(path, 'r', encoding='utf-8')
        data = np.empty(shape=[0, 3])
        cnt_i = 0
        
        #对test文件夹里的文件进行循环操作，即对每位出租车司机的数据进行操作，把数据存储到data变量中
        for line in file.readlines():
            spt = line.split(",")
            data = np.append(data, [[float(spt[0]), float(spt[1]), int(spt[2])]], axis=0)
            cnt_i += 1
            #print(spt[0],spt[1],spt[2],spt[3])
            
        #按照时间排序
        for i in range(0,cnt_i):
            for j in range(i+1,cnt_i):
                #print("i="+str(i)+"j="+str(j)+"\n")
                if data[j,2] < data[i,2]:
                    #print("data[i,2]="+str(data[i,2])+'\n')
                    #print("data[j,2]="+str(data[j,2])+'\n')
                    (data[i,0],data[j,0]) = exchange(float(data[i,0]),float(data[j,0]))
                    (data[i,1],data[j,1]) = exchange(float(data[i,1]),float(data[j,1]))
                    (data[i,2],data[j,2]) = exchange(int(data[i,2]),int(data[j,2]))
                    
        #计算通量
#        for i in range(0,cnt_i-1):
#            #point(data[i,0],data[i,1]),point(data[i+1,0],data[i+1,1])
#            #road:116.39411442300	39.76129007620
#            #road:(116.37411442300,39.76129007620,116.41411442300,39.76129007620)
#            a = segment_point(115.17411442300,39.76129007620,117.61411442300,39.76129007620,data[i,0],data[i,1],data[i+1,0],data[i+1,1])
#            #print(a)
#            #print(segment_point(115.17411442300,39.76129007620,117.61411442300,39.76129007620,116.384114423,39.76029007620,116.394114423,39.76329007620))
#            if a == 1:
#                sum1 = sum1+1
#            elif a == -1:
#                sum2 = sum2+1
#            elif a == 2:
#                sum3 = sum3+1
#            elif a == -2:
#                sum4 = sum4+1
#            else:
#                continue
        
        horizontal = open("E:\\88-滴滴数据new\\1-data\\horizontal.txt", 'r', encoding='utf-8')
        for line_h in horizontal.readlines():
            spt_h = line_h.split(",")
            sum1 = 0
            sum2 = 0
            for i in range(0,cnt_i-1):
                #point(data[i,0],data[i,1]),point(data[i+1,0],data[i+1,1])
                #road:116.39411442300	39.76129007620
                a = segment_point(float(spt_h[1])-0.00117,float(spt_h[2]),float(spt_h[1])+0.00117,float(spt_h[2]),data[i,0],data[i,1],data[i+1,0],data[i+1,1])
                #print(a)
                #print(segment_point(115.17411442300,39.76129007620,117.61411442300,39.76129007620,116.384114423,39.76029007620,116.394114423,39.76329007620))
                if a == 1:
                    sum1 = sum1+1
                elif a == -1:
                    sum2 = sum2+1
                else:
                    continue
            #程序结果写入到file_out里
        file_out.write(spt_h[0]+","+str(sum1)+","+str(sum2)+"\n")
        vertical = open("E:\\88-滴滴数据new\\1-data\\vertical.txt", 'r', encoding='utf-8')
        for line_h in vertical.readlines():
            spt_h = line_h.split(",")
            sum3 = 0
            sum4 = 0
            for i in range(0,cnt_i-1):
                #point(data[i,0],data[i,1]),point(data[i+1,0],data[i+1,1])
                #road:116.39411442300	39.76129007620
                a = segment_point(float(spt_h[1]),float(spt_h[2])-0.000899,float(spt_h[1]),float(spt_h[2])+0.000899,data[i,0],data[i,1],data[i+1,0],data[i+1,1])
                #print(a)
                #print(segment_point(115.17411442300,39.76129007620,117.61411442300,39.76129007620,116.384114423,39.76029007620,116.394114423,39.76329007620))
                if a == 2:
                    sum3 = sum3+1
                elif a == -2:
                    sum4 = sum4+1
                else:
                    continue
            #程序结果写入到file_out里
        file_out1.write(spt_h[0]+","+str(sum3)+","+str(sum4)+"\n")
    #关闭文件
    file_out.close()
    file_out1.close()
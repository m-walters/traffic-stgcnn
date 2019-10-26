# -*- coding: utf-8 -*-
"""
These codes are used to divide Beijing into squres and identify roads(directions) in each square.
"""

import numpy as np
import matplotlib.colors
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

def genMatrix(rows,cols): 
    """
    Define a matrix.
    """ 
    matrix = np.empty(shape=[rows, cols])
    return matrix

if __name__ == "__main__":
    
    longitude = 0.011741652782473 #How many degrees is 1km equal to in longitude.
    latitude = 0.008994627867046 #How many degrees is 1km equal to in latitude.
    
    #1km^2 square  0.011741653/2 0.0089946279/2  33*33
    Xnum = 33 #Number of squres:
    Ynum = 33
    Xstep = 0.011741653 #Length of side in degree of two directions.
    Ystep = 0.0089946279
    X0 = 116.18995243500 - Xstep/2 #Origin of coordinates.
    Y0 = 39.75679276210 - Ystep/2
    file_out_angle_x = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_x_1000.txt",'w')
    file_out_angle_y = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_y_1000.txt",'w')
    file_out_num = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_num_1000.txt",'w')
    
#    #0.5km^2 square  0.005870826/2 0.0044973139/2  67*66
#    Xnum = 67
#    Ynum = 66
#    Xstep = 0.005870826
#    Ystep = 0.0044973139
#    X0 = 116.18701702200 - Xstep/2
#    Y0 = 39.75454410520 - Ystep/2
#    file_out_angle_x = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_x_500.txt",'w')
#    file_out_angle_y = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_y_500.txt",'w')
#    file_out_num = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/result_num_500.txt",'w')
    
#    #0.2km^2 square  0.00234833/2 0.0017989256/2  166*165
#    Xnum = 166
#    Ynum = 165
#    Xstep = 0.00234833
#    Ystep = 0.0017989256
#    X0 = 116.18525577400 - Xstep/2
#    Y0 = 39.75319491100 - Ystep/2
#    file_out_angle_x = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_x_200.txt",'w')
#    file_out_angle_y = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result_angle_y_200.txt",'w')
#    file_out_num = open("/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/result_num_200.txt",'w')

    path_dir = "/home/jiayu/pythonfile/Artificial Intellegence/Prediction of Congestion/files/result/"    
    all_dir = os.listdir(path_dir)
    cnt = 0 #Used to count the number of files. 
    #We initialize that there are 100000 directions roads in each square.
    #Xnum*Ynum is the total number of squres.
    data_angle_x = genMatrix(Xnum*Ynum,100000) 
    data_angle_y = genMatrix(Xnum*Ynum,100000)
    data_cnt = np.zeros([Xnum*Ynum,1]) #Used to count how many direactions in each square.
    for a_file in all_dir:
        path = path_dir + "%s"%a_file
        file = open(path, 'r')
        cnt += 1
        data = genMatrix(0, 4) #data : [[lon],[lat],[dt],[day]]
        cnt_i = 0 #Used to count the number of lines in each file.
        print("processing batch "+str(cnt)+"..."+"file name '%s'"%a_file)
        lines = file.readlines()
        del(lines[0])
        for line in lines:
            the_line = line.split(", ")
            cnt_i += 1
            data = np.append(data, [[the_line[0], the_line[1], the_line[3], the_line[-1]]], axis=0)

        for i in range(1,cnt_i-1):
            deltaX = (float(data[i+1,0])-float(data[i,0]))/longitude
            deltaY = (float(data[i+1,1])-float(data[i,1]))/latitude
            
            #Ignore those who didn't move.
            if deltaX == 0 and deltaY == 0:
                continue
            
            #Ignore those points whose interval with its former point is longer than 1min.
            Time = float(data[i][2])/3600000
            if Time > 1.0/60 or Time == 0:
                continue   

            #Calculate velocity of x and y direction respectively.
            v_x = deltaX/Time
            v_y = deltaY/Time
            
            id_iX = math.floor( (float(data[i,0]) - X0)/Xstep )
            id_iY = math.floor( (float(data[i,1]) - Y0)/Ystep )
            id_iNextX = math.floor( (float(data[i+1,0]) - X0)/Xstep )
            id_iNextY = math.floor( (float(data[i+1,1]) - Y0)/Ystep )
            id_iCenterX = math.floor( ((float(data[i+1,0])+float(data[i,0]))/2 - X0)/Xstep )
            id_iCenterY = math.floor( ((float(data[i+1,1])+float(data[i,1]))/2 - Y0)/Ystep )
            
            if id_iX >= 0 and id_iX <= Xnum-1 and id_iY >= 0 and id_iY <= Ynum-1:
                id_i = id_iX + id_iY*Xnum
                data_angle_x[ id_i ][ data_cnt[id_i][0] ] = v_x
                data_angle_y[ id_i ][ data_cnt[id_i][0] ] = v_y
                data_cnt[id_i][0] = data_cnt[int(id_i)][0]+1
            if id_iNextX >= 0 and id_iNextX <= Xnum-1 and id_iNextY >= 0 and id_iNextY <= Ynum-1:
                id_iNext = id_iNextX + id_iNextY*Xnum
                data_angle_x[ id_iNext ][ data_cnt[id_iNext][0] ] = v_x
                data_angle_y[ id_iNext ][ data_cnt[id_iNext][0] ] = v_y
                data_cnt[id_iNext][0] = data_cnt[int(id_iNext)][0]+1
            if id_iCenterX >= 0 and id_iCenterX <= Xnum-1 and id_iCenterY >= 0 and id_iCenterY <= Ynum-1:
                id_iCenter = id_iCenterX + id_iCenterY*Xnum
                data_angle_x[ id_iCenter ][ data_cnt[id_iCenter][0] ] = v_x
                data_angle_y[ id_iCenter ][ data_cnt[id_iCenter][0] ] = v_y
                data_cnt[id_iCenter][0] = data_cnt[int(id_iCenter)][0]+1
    

            
            
    

            
    #Cluster
#    eps = 10
#    min_samples = 10
#    model = DBSCAN(eps=eps, min_samples=min_samples)
#    model.fit(data_angle_x[0])
#    y_hat = model.labels_
#    core_indices = np.zeros_like(y_hat, dtype=bool)
#    core_indices[model.core_sample_indices_] = True
#    
#    y_unique = np.unique(y_hat)
#    n_clusters = y_unique.size - (1 if -1 in y_hat else 0)
    
    
    #Write the result in files.
    for i in range(0,Xnum*Ynum-1):
        file_out_angle_x.write(str(i))
        file_out_angle_y.write(str(i))
        file_out_num.write(str(i))
        for j in range(0,int(data_cnt[i][0])-1):
            file_out_angle_x.write(","+str(data_angle_x[i][j]))
            file_out_angle_y.write(","+str(data_angle_y[i][j]))
        file_out_angle_x.write("\n")
        file_out_angle_y.write("\n")
        file_out_num.write(","+str(int(data_cnt[i][0]))+"\n")
    #Close files.
    file_out_angle_x.close()
    file_out_angle_y.close()
    file_out_num.close()
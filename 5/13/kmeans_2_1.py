import numpy as np
import math

def Eucl_Dist(X,Y):
    d = math.sqrt(sum([(x-y)**2 for (x,y) in zip(X,Y)]))
    return d

center = np.array([[12,15,13,28,24],[7,11,10,19,21],[12,14,11,27,23]])
data = np.array([[12,15,13,28,24],[7,11,10,19,21],[12,14,11,27,23],[6,7,4,13,20],[13,14,13,27,25]])
labels = np.zeros((3,5))
def Kmeans(centers):
    for i in range(0,5):
        x = Eucl_Dist(centers[0],data[i])
        y = Eucl_Dist(centers[1], data[i])
        z = Eucl_Dist(centers[2], data[i])
        if x<y and x<z:
            labels[0,i] = 1
            labels[1,i] = 0
            labels[2,i] = 0
        elif y<x and y<z:
            labels[0, i] = 0
            labels[1, i] = 1
            labels[2, i] = 0
        elif z<x and z<y:
            labels[0, i] = 0
            labels[1, i] = 0
            labels[2, i] = 1
    num = np.array([0,0,0])
    sum = np.zeros((3,5))
    for i in range(0,5):
        if labels[0,i] == 1:
            num [0] = num[0] + 1
            sum[0][0] = sum[0][0] + data[i][0]
            sum[0][1] = sum[0][1] + data[i][1]
            sum[0][2] = sum[0][2] + data[i][2]
            sum[0][3] = sum[0][3] + data[i][3]
            sum[0][4] = sum[0][4] + data[i][4]
        elif labels[1,i] == 1:
            num[1] = num[1] + 1
            sum[1][0] = sum[1][0] + data[i][0]
            sum[1][1] = sum[1][1] + data[i][1]
            sum[1][2] = sum[1][2] + data[i][2]
            sum[1][3] = sum[1][3] + data[i][3]
            sum[1][4] = sum[1][4] + data[i][4]
        else:
            num[2] = num[2] + 1
            sum[2][0] = sum[2][0] + data[i][0]
            sum[2][1] = sum[2][1] + data[i][1]
            sum[2][2] = sum[2][2] + data[i][2]
            sum[2][3] = sum[2][3] + data[i][3]
            sum[2][4] = sum[2][4] + data[i][4]
    new_center = np.zeros((3,5))
    for i in range(0,3):
        new_center[i] = sum[i]/num[i]
    diff = new_center - centers
    center = new_center
    if diff[0, 0] == 0 and diff[0, 1] == 0 and diff[1, 0] == 0 and diff[1, 1] == 0 and diff[2, 0] == 0 and diff[2, 1] == 0:
        print (center)
        return center,labels
    else:
        return Kmeans(center)

center,labels=Kmeans(center)
print (labels)
import numpy as np
import math

#定义欧式距离
def Eucl_Dist(X,Y):
    d = math.sqrt(sum([(x-y)**2 for (x,y) in zip(X,Y)]))
    return d

center = np.array([[1,1],[2,1]])  #选择初始聚类中心
data = np.array([[1,1],[2,1],[4,3],[5,4]])
labels = np.zeros((2,4))#记录到中心的距离和标签

#kmeans算法递归更新中心点
def Kmeans(centers):
    for i in range(0,4):
        x = Eucl_Dist(centers[0],data[i])
        y = Eucl_Dist(centers[1],data[i])
        if x>y:
            labels[0,i] = 0
            labels[1,i] = 1
        else:
            labels[0,i] = 1
            labels[1,i] = 0
    num = np.array([0,0])
    sum = np.array([[0,0],[0,0]])
    for i in range(0,4):
        if labels[0,i] == 1:
            num[0] = num[0] + 1
            sum[0][0] = sum[0][0] + data[i][0]
            sum[0][1] = sum[0][1] + data[i][1]
        else:
            num[1] = num[1] + 1
            sum[1][0] = sum[1][0] + data[i][0]
            sum[1][1] = sum[1][1] + data[i][1]
    new_center = np.zeros((2,2))
    for i in range(0,2):
        new_center[i] = sum[i]/num[i]
    diff = new_center - centers
    if diff[0,0] == 0 and diff[0,1] == 0 and diff[1,0] == 0 and diff[1,1] == 0:
        center = new_center
        return center,labels
    else:
        center = new_center
        print (center)
        return Kmeans(center)

center,labels=Kmeans(center)
print (labels)

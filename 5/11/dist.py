from numpy import *
import math

def eucliDist(A,B):
    return math.sqrt(sum([(a-b)**2 for (a,b) in zip(A,B)]))
X=[2,2,3]
Y=[1,1,2]
print(eucliDist(X,Y))

vector1 = mat([1, 2, 3])
vector2 = mat([4, 5, 6])

man_dis = sum(abs(vector1 - vector2))
print(man_dis)

che_dis = abs(vector1-vector2).max()
print(che_dis)


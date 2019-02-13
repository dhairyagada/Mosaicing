import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
from myconfig import *
colimg = cv2.imread('./NeuralNet/SampleTestSet/COCO_test2014_000000000001.jpg')
colimg = cv2.resize(colimg,(w,h))
img = cv2.cvtColor(colimg,cv2.COLOR_RGB2GRAY)

rho = 32
patchsize = 128

x = random.randint(rho,x_l-rho-patchsize)
y = random.randint(rho,h-rho-patchsize)
cv2.rectangle(colimg,(rho,rho),(x_l-rho-patchsize,h-rho-patchsize),(0,255,255),2)


upleft = (x,y)
botleft = (x,y+patchsize)
botright = (x+patchsize,y+patchsize)
upright = (x+patchsize,y)

points = [upleft,botleft,botright,upright]

perturbedpoints = []
for point in points:
    perturbedpoints.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))



## Drawing
points = np.array(points)
points = points.reshape((1,4,2))
colimg = cv2.polylines(colimg,points,1,(0,0,255),2)

perturbedpoints = np.array(perturbedpoints)
perturbedpoints = perturbedpoints.reshape((1,4,2))
colimg = cv2.polylines(colimg,perturbedpoints,1,(255,0,0),2)


cv2.imshow("img",colimg)
k =cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
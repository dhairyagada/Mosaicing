import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
from mynetconfig import *
from glob import glob
import os

colimg = cv2.imread('./NeuralNet/SampleTestSet/COCO_test2014_000000000014.jpg')
colimg = cv2.resize(colimg,(w,h))
img = cv2.cvtColor(colimg,cv2.COLOR_RGB2GRAY)

x = random.randint(rho,x_l-rho-patchsize)
y = random.randint(rho,h-rho-patchsize-newpointdel)
cv2.rectangle(colimg,(rho,rho),(x_l-rho-patchsize,h-rho-patchsize),(0,255,255),2)


upleft = (x,y)
botleft = (x,y+patchsize)
botright = (x+patchsize,y+patchsize)
upright = (x+patchsize,y)

points = [upleft,botleft,botright,upright]

perturbedpoints = []
for point in points:
    perturbedpoints.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

newx = x + random.randint(1,newpointdel)
newy = y + random.randint(1,newpointdel)

newupleft = (newx,newy)
newbotleft = (newx,newy+patchsize)
newbotright = (newx+patchsize,newy+patchsize)
newupright = (newx+patchsize,newy)

newpoints = [newupleft,newbotleft,newbotright,newupright]


## Drawing
points = np.array(points)
points = points.reshape((1,4,2))
colimg = cv2.polylines(colimg,points,1,(0,0,255),2)

perturbedpoints = np.array(perturbedpoints)
perturbedpoints = perturbedpoints.reshape((1,4,2))
colimg = cv2.polylines(colimg,perturbedpoints,1,(255,0,0),2)

newpoints = np.array(newpoints)
newpoints = newpoints.reshape((1,4,2))
colimg = cv2.polylines(colimg,newpoints,1,(255,0,255),2)

perturbedpoints = np.float32(perturbedpoints)
newpoints = np.float32(newpoints)

HTrain = cv2.getPerspectiveTransform(newpoints,perturbedpoints)
HTraininv = inv(HTrain)
print('H = ',HTrain)

warpedimage  = cv2.warpPerspective(colimg,HTraininv,(w,h))

original_patch = colimg[newy:newy + patchsize, newx:newx + patchsize]
warped_patch = warpedimage[newy:newy + patchsize, newx:newx + patchsize]


cv2.imshow("img",colimg)
cv2.imshow("warped",warpedimage)

cv2.imshow("OgPatch",original_patch)
cv2.imshow("WarpedPatch",warped_patch)
k =cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
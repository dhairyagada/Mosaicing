from __future__ import print_function
import Registration
import matplotlib.pyplot as plt
from util import *
import cv2
import numpy as np
from time import sleep
from ImageProc.cylinderwarp import *
from myconfig import *
from ImageProc.PreProcessing import *

def show(imgX):
    plt.plot()
    plt.imshow(cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB))
    plt.show()
    return


def NetRunHom(IX,IY):
    IX = cv2.resize(IX,(w,h))
    IY = cv2.resize(IY,(w,h))
    #IX = cylindricalWarp(IX)
    #IY = cylindricalWarp(IY)
    KpX = IX.copy()
    KpY = IY.copy()

    matches = np.zeros((IX.shape[0],IX.shape[1]*2,3))

    matches[:IX.shape[0],:IX.shape[1]] = KpX.copy()
    matches[:IX.shape[0],IX.shape[1]:IX.shape[1]*2] = KpY.copy() 
    
    IX1 = Hist_Eq(IX,clippinglimit)
    IY1 = Hist_Eq(IY,clippinglimit)

    reg = Registration.CNN()
    #register
    X, Y, Z = reg.register(IX1, IY1)

    Xdash = X.copy()
    Ydash = Y.copy()
    Zdash = Z.copy()

    

    for j in range(len(X)):
        Xdash[j,1] = X[j,0]
        Xdash[j,0] = X[j,1]

        Ydash[j,1] = Y[j,0]
        Ydash[j,0] = Y[j,1]

        Zdash[j,1] = Z[j,0]
        Zdash[j,0] = Z[j,1]

    for i in range(len(X)):
        cv2.circle(KpX,(int(X[i,1]),int(X[i,0])),3, (0,0,255), 2)
        cv2.circle(KpY,(int(Y[i,1]),int(Y[i,0])),3, (0,0,255), 2)

        cv2.circle(matches,(int(X[i,1]),int(X[i,0])),3, (0,0,255), 2)
        cv2.circle(matches,(int(Y[i,1])+IX.shape[1],int(Y[i,0])),3, (0,0,255), 2)

        cv2.line(matches,(int(X[i,1]),int(X[i,0])),(int(Y[i,1])+IX.shape[1],int(Y[i,0])),(0,255,255),2)

    points1 = np.float32(Zdash).reshape(-1,1,2)
    points2 = np.float32(Ydash).reshape(-1,1,2)

    Hom, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    Warped_Img = cv2.warpPerspective(IY, Hom, (int(3*w),int(1.2*h)))

    FinalImage = Warped_Img.copy()
    FinalImage[0:IX.shape[0], 0:IX.shape[1]] = IX

    return KpX,KpY,matches,Hom,Warped_Img,FinalImage
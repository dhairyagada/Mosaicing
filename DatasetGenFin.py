import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
from mynetconfig import *
from glob import glob
import os
from pylab import *
import pickle

def show(im):
    plt.plot()
    plt.imshow(im)
    plt.show()
    return

loc_list = glob(rawpdatapath)

X = np.zeros((datalen, 128, 128, 2))  # images
Y = np.zeros((datalen,8))

for i in range(datalen):
    
    # Only For Selecting Landscape Images!
    while True:  
        index = randint(0,numrawimages-1)
        colimgloc = loc_list[index]
        colimg = cv2.imread(colimgloc)
        print(colimg.shape[0:2])
        if(1.9*colimg.shape[0]>colimg.shape[1] & colimg.shape[1] > 1.5*colimg.shape[0]-1 ):  
            break  
    
    
    img = cv2.cvtColor(colimg,cv2.COLOR_RGB2GRAY)
    deloverlap = randint(0,lapdisp)

    leftpatch = colimg[0:colimg.shape[0],0:colimg.shape[0]]
    rightpatch = colimg[0+rho:colimg.shape[0]-rho,colimg.shape[1]-colimg.shape[0]-deloverlap+rho:colimg.shape[1]-deloverlap-rho]

    ## Actual Points For Right Image
    topleft = (colimg.shape[1]-colimg.shape[0]-deloverlap+rho,0+rho)                    # Yellow
    bottomleft = (colimg.shape[1]-colimg.shape[0]-deloverlap+rho,colimg.shape[0]-rho)   # Orange
    bottomright = (colimg.shape[1]-deloverlap-rho,colimg.shape[0]-rho)                  #LightBlue
    topright = (colimg.shape[1]-deloverlap-rho,0+rho)                                   #Purple

    actual_points = [topleft,bottomleft,bottomright,topright]

    ap = actual_points.copy()


    ## Perturbed Points For Perspective
    perturbedpoints = []
    for point in actual_points:
        perturbedpoints.append((point[0] + randint(-rho, rho), point[1] + randint(-rho, rho)))

    pp = perturbedpoints.copy()
    perturbedpoints = np.float32(perturbedpoints)
    actual_points = np.float32(actual_points)


    # Perspective and Warping
    HTrain = cv2.getPerspectiveTransform(actual_points,perturbedpoints)
    HTraininv = inv(HTrain)
    #print('H = ',HTrain)
    warpedimage = cv2.warpPerspective(colimg,HTraininv,(800,800))

    rightpatchwarped = warpedimage[0+rho:colimg.shape[0]-rho,colimg.shape[1]-colimg.shape[0]-deloverlap+rho:colimg.shape[1]-deloverlap-rho]

    if(datavis == 1):
        
        cv2.rectangle(colimg,(0,0),(colimg.shape[0],colimg.shape[0]),(0,255,0),2)
        cv2.rectangle(colimg,(colimg.shape[1]-colimg.shape[0]-deloverlap+rho,0+rho),(colimg.shape[1]-deloverlap-rho,colimg.shape[0]-rho),(0,0,255),2)
        cv2.circle(colimg,topleft, 10, (255,255,0), -1)
        cv2.circle(colimg,bottomleft,10,(255,128,0), -1)
        cv2.circle(colimg,bottomright,10,(0,255,128),-1)
        cv2.circle(colimg,topright,10,(102,0,102),-1)

        ap = np.array(ap)
        ap = ap.reshape((1,4,2))
        
        print("Perturbed Points = ",perturbedpoints)
        print("H = ",HTraininv)

        show(colimg)
        show(leftpatch)
        #show(rightpatch)
        #show(rightpatchwarped)
        show(cv2.polylines(warpedimage,ap,1,(102,0,102),3))
        plt.subplot(1,2,1)
        plt.imshow(rightpatch)
        plt.subplot(1,2,2)
        plt.imshow(rightpatchwarped)
        plt.show()

    ## Converting Patches to GrayScale
    leftpatch = cv2.cvtColor(leftpatch,cv2.COLOR_RGB2GRAY)
    rightpatchwarped = cv2.cvtColor(rightpatchwarped,cv2.COLOR_RGB2GRAY)

    if(datavis == 1):
        plt.subplot(1,2,1)
        plt.imshow(leftpatch)
        plt.subplot(1,2,2)
        plt.imshow(rightpatchwarped)
        plt.show()

    ## to 128 bit patches

    leftpatch = cv2.resize(leftpatch,(128,128))
    rightpatchwarped = cv2.resize(rightpatchwarped,(128,128))

    if(datavis == 1):
        plt.subplot(1,2,1)
        plt.imshow(leftpatch)
        plt.subplot(1,2,2)
        plt.imshow(rightpatchwarped)
        plt.show()

    ##
    actual_points_arr = np.array(actual_points)
    perturbedpoints_arr = np.array(perturbedpoints)

    trainingimage = np.dstack((leftpatch,rightpatchwarped))
    H_four_points = np.subtract(perturbedpoints_arr, actual_points_arr)
    
    X[i,:,:,:] = trainingimage
    Y[i,:] = H_four_points.reshape(-1)
    
    print('Progress :[%f%%] \r'%(i/100),end="")

print("Saving in pickle format.")
X_train = X[0:int(0.9 *datalen)]
X_valid = X[int(0.9 *datalen):]
Y_train = Y[0:int(0.9 *datalen)]
Y_valid = Y[int(0.9 *datalen):]
train = {'features': X_train, 'labels': Y_train}
valid = {'features': X_valid, 'labels': Y_valid}
pickle.dump(train, open("./train2.p", "wb"))
pickle.dump(valid, open("./valid2.p", "wb"))
print("Done.")

k =cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
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
        if(colimg.shape[1] > 1.5*colimg.shape[0]-1):  
            break  
    
    
    img = cv2.cvtColor(colimg,cv2.COLOR_RGB2GRAY)
    deloverlap = randint(0,lapdisp)

    leftpatch = img[0:colimg.shape[0],0:colimg.shape[0]]
    rightpatch = img[0:colimg.shape[0],colimg.shape[1]-colimg.shape[0]-deloverlap:colimg.shape[1]-deloverlap]

    ## Actual Points For Right Image
    topleft = (colimg.shape[1]-colimg.shape[0]-deloverlap,0)                    # Yellow
    bottomleft = (colimg.shape[1]-colimg.shape[0]-deloverlap,colimg.shape[0])   # Orange
    bottomright = (colimg.shape[1]-deloverlap,colimg.shape[0])                  #LightBlue
    topright = (colimg.shape[1]-deloverlap,0)                                   #Purple

    actual_points = [topleft,bottomleft,bottomright,topright]

    perturbedpoints = []
    for point in actual_points:
        perturbedpoints.append((point[0] + randint(-rho, rho), point[1] + randint(-rho, rho)))


    perturbedpoints = np.float32(perturbedpoints)
    actual_points = np.float32(actual_points)

    HTrain = cv2.getPerspectiveTransform(actual_points,perturbedpoints)
    HTraininv = inv(HTrain)
    #print('H = ',HTrain)

    warpedimage = cv2.warpPerspective(colimg,HTraininv,(800,800))

    warpedpatchright = warpedimage[0:colimg.shape[0],colimg.shape[1]-colimg.shape[0]-deloverlap:colimg.shape[1]-deloverlap]
    if(datavis == 1):
        
        cv2.rectangle(colimg,(0,0),(colimg.shape[0],colimg.shape[0]),(0,255,0),2)
        cv2.rectangle(colimg,(colimg.shape[1]-colimg.shape[0]-deloverlap,0),(colimg.shape[1]-deloverlap,colimg.shape[0]),(0,0,255),2)
        cv2.circle(colimg,topleft, 10, (255,255,0), -1)
        cv2.circle(colimg,bottomleft,10,(255,128,0), -1)
        cv2.circle(colimg,bottomright,10,(0,255,128),-1)
        cv2.circle(colimg,topright,10,(102,0,102),-1)

        print("Perturbed Points = ",perturbedpoints)
        print("H = ",HTraininv)
        #show(kpis)
        show(colimg)
        show(leftpatch)
        #show(rightpatch)
        #show(warpedpatchright)
        show(warpedimage)
        plt.subplot(1,2,1)
        plt.imshow(rightpatch)
        plt.subplot(1,2,2)
        plt.imshow(warpedpatchright)
        plt.show()

    continue
    x = randint(rho,x_l-rho-patchsize)
    y = randint(rho,h-rho-patchsize-newpointdel)
    cv2.rectangle(colimg,(rho,rho),(x_l-rho-patchsize,h-rho-patchsize),(0,255,255),2)


    upleft = (x,y)
    botleft = (x,y+patchsize)
    botright = (x+patchsize,y+patchsize)
    upright = (x+patchsize,y)

    points = [upleft,botleft,botright,upright]

    perturbedpoints = []
    for point in points:
        perturbedpoints.append((point[0] + randint(-rho, rho), point[1] + randint(-rho, rho)))

    perturbedpoints_arr = np.array(perturbedpoints)  # For Shape 8

    newx = x + randint(1,newpointdel)
    newy = y + randint(1,newpointdel)

    newupleft = (newx,newy)
    newbotleft = (newx,newy+patchsize)
    newbotright = (newx+patchsize,newy+patchsize)
    newupright = (newx+patchsize,newy)

    newpoints = [newupleft,newbotleft,newbotright,newupright]
    newpoints_arr = np.array(newpoints)

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
    #print('H = ',HTrain)

    warpedimage  = cv2.warpPerspective(colimg,HTraininv,(w,h))

    original_patch = colimg[newy:newy + patchsize, newx:newx + patchsize]
    warped_patch = warpedimage[newy:newy + patchsize, newx:newx + patchsize]

    oggraypatch = cv2.cvtColor(original_patch,cv2.COLOR_RGB2GRAY)
    warpedgraypatch = cv2.cvtColor(warped_patch,cv2.COLOR_RGB2GRAY)

    trainingimage = np.dstack((oggraypatch,warpedgraypatch))
    H_four_points = np.subtract(perturbedpoints_arr, newpoints_arr)

    X[i,:,:,:] = trainingimage
    Y[i,:] = H_four_points.reshape(-1)
    
    print('Progress :[%f%%] \r'%(i/100),end="")
    
    """ subplot(2,2,1)
    plt.imshow(colimg)
    subplot(2,2,2)
    plt.imshow(warpedimage)
    subplot(2,2,3)
    plt.imshow(original_patch)
    subplot(2,2,4)
    plt.imshow(warped_patch)
    plt.show() """

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
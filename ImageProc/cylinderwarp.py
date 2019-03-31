from myconfig import *
import numpy as np
import cv2
from matplotlib import pyplot as plt


def cylindricalWarp(img):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    f = focal_len
    h,w = img.shape[:2]
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix

    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    cylimg = cv2.cvtColor(cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA),cv2.COLOR_BGRA2BGR)

    count1 = 0
    i = cylimg.shape[1] -1
    
    count0 = 0
    j = cylimg.shape[0] -1

    while ((cylimg[int((cylimg.shape[0])/2),i] == [0,0,0]).all()):
        count1 = count1 +1
        i = i - 1

    
    cylimg = cylimg[:cylimg.shape[0],count1:cylimg.shape[1]-count1]

    while ((cylimg[j,cylimg.shape[1]-1] == [0,0,0]).all()):
        count0 = count0 +1
        j = j - 1
    
    cylimg = cylimg[count0:cylimg.shape[0]-count0,:cylimg.shape[1]]
    return cylimg

""" 
def show(im):
    plt.plot()
    plt.imshow(im)
    plt.show()
    return

show(cylindricalWarp(cv2.imread(img1))) """
from myconfig import *
import cv2
import numpy as np

def SIFT(imgl,imgr):

    sift=cv2.xfeatures2d.SIFT_create()

    if ifROI == 0:
        kpl,descl = sift.detectAndCompute(imgl,None)
        kpr,descr = sift.detectAndCompute(imgr,None)
    else:
        kpl,descl = sift.detectAndCompute(imgl[y_l:yw_l,x_l:xw_l],None)
        kpr,descr = sift.detectAndCompute(imgr[y_r:yw_r,x_r:xw_r],None)
        
        for i in range(len(kpl)):
            kpl[i].pt = (kpl[i].pt[0] + float(x_l),kpl[i].pt[1])

    
    kpImageLeft =cv2.drawKeypoints(imgl,kpl,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpImageRight =cv2.drawKeypoints(imgr,kpr,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return kpl,kpr,descl,descr,kpImageLeft,kpImageRight

def Matcher(imgl,imgr,iml_kp,imr_kp,iml_desc,imr_desc):
    
    bf = cv2.BFMatcher(cv2.NORM_L2)

    matches = bf.knnMatch(iml_desc, imr_desc, k=2)

    good = []
    goodT= []

    for m,n in matches:
        if m.distance < min_limit*n.distance:
            good.append([m])
            goodT.append(m)

    if ROIDisp==1 & ifROI==1:
        cv2.rectangle(imgl,(x_l,y_l),(xw_l,yw_l),(0,255,0),2)
        cv2.rectangle(imgr,(x_r,y_r),(xw_r,yw_r),(0,255,0),2)

    match_img = cv2.drawMatchesKnn(imgl,iml_kp,imgr,imr_kp,good,None,flags=2)

    return goodT,match_img

def KeypointMatcher(imageLeft,imageRight):

    kpl,kpr,descl,descr,kpImageLeft,kpImageRight = SIFT(imageLeft,imageRight)

    goodT,match_img = Matcher(imageLeft,imageRight,kpl,kpr,descl,descr)

    return kpl,kpr,descl,descr,kpImageLeft,kpImageRight,goodT,match_img
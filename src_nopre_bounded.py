import cv2
import numpy as np

img1='InputImages/7.jpeg'
img2='InputImages/8.jpeg'
Name_Final = 'OutputImages/Op4WithoutPre_Bounded.jpeg'

w=360
h=480


# Region of Interest

## Left Image
x_l     = 200
xw_l    = 360

y_l     =  0
yw_l    =  480

## Right Image
x_r     = 0
xw_r    = 160

y_r     = 0
yw_r    = 480

min_limit=0.75

def inp(A,B):
    img1=cv2.imread(A)
    img2=cv2.imread(B)
    return img1,img2

def Resize_BW(img,w,h):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, (w,h))
    return img

def SIFT(imgl,imgr):
    sift=cv2.xfeatures2d.SIFT_create()
    kpl,descl = sift.detectAndCompute(imgl[y_l:yw_l,x_l:xw_l],None)
    img_keypl =cv2.drawKeypoints(imgl,kpl,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    kpr,descr = sift.detectAndCompute(imgr[y_r:yw_r,x_r:xw_r],None)
    img_keypr =cv2.drawKeypoints(imgr,kpr,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for i in range(len(kpl)):
        kpl[i].pt = (kpl[i].pt[0] + 200.0,kpl[i].pt[1])
    
    return kpl,descl,img_keypl,kpr,descr,img_keypr

def Matcher(ima,imb,ima_kp,imb_kp,ima_desc,imb_desc,min_limit):
    bf = cv2.BFMatcher(cv2.NORM_L2)

    matches = bf.knnMatch(ima_desc, imb_desc, k=2)
    good = []
    for m,n in matches:
        if m.distance < min_limit*n.distance:
            good.append([m])
    cv2.rectangle(ima,(x_l,y_l),(xw_l,yw_l),(0,255,0),2)
    cv2.rectangle(imb,(x_r,y_r),(xw_r,yw_r),(0,255,0),2)
    match_img = cv2.drawMatchesKnn(ima,ima_kp,imb,imb_kp,good,None,flags=2)

    return matches,good,match_img

def FinalCall(A,B,min_limit):
    imBWA = Resize_BW(A,w,h)
    imBWB = Resize_BW(B,w,h)
    
    ima_kp,ima_desc,keyp_imgA,imb_kp,imb_desc,keyp_imgB = SIFT(imBWA,imBWB)

    matches,good,match_img = Matcher(imBWA,imBWB,ima_kp,imb_kp,ima_desc,imb_desc,min_limit)

    return match_img

imgA,imgB = inp(img1,img2)

finalimg=FinalCall(imgA,imgB,min_limit)

cv2.imshow('Final',finalimg )
cv2.imwrite(Name_Final,finalimg)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

import cv2
import numpy as np

img1='InputImages/4.JPG'
img2='InputImages/3.JPG'
Name_Final = 'OutputImages/Op2AlignedImg.jpeg'
downsample_level=1
clippinglimit=2
w=360
h=480
gaussian_ksize=3
min_limit=0.75

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


def inp(A,B):
    img1=cv2.imread(A)
    img2=cv2.imread(B)
    return img1,img2

def Hist_Eq(img,climit):
    image_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    merged_channels = cv2.merge((cl, a_channel, b_channel))
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image

def Resize_BW(img,w,h):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, (w,h))
    return img

def Downsample(img,downlevel,w,h):
    xrange=range
    G = img
    for i in xrange(downlevel):
        G = cv2.pyrDown(G)
    G=cv2.resize(G,(w,h))
    return G

def GaussianLPF(img,k_size):
    img=cv2.GaussianBlur(img,(k_size,k_size),0)
    return img

def prep(img,w,h,climit,downsample_level,k_size):
    img1=Downsample(img,downsample_level,w,h)
    img2=Hist_Eq(img1,climit)
    img3=Resize_BW(img2,w,h)
    img4=GaussianLPF(img3,k_size)

    return img1,img2,img3,img4

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
    goodT= []
    for m,n in matches:
        if m.distance < min_limit*n.distance:
            good.append([m])
            goodT.append(m)
    cv2.rectangle(ima,(x_l,y_l),(xw_l,yw_l),(0,255,0),2)
    cv2.rectangle(imb,(x_r,y_r),(xw_r,yw_r),(0,255,0),2)
    match_img = cv2.drawMatchesKnn(ima,ima_kp,imb,imb_kp,good,None,flags=2)

    return matches,good,match_img,goodT

def AlignImages(ima,imb,kp1,kp2,good):
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    points1 = np.float32([ (kp1[m.queryIdx]).pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ (kp2[m.trainIdx]).pt for m in good ]).reshape(-1,1,2)

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    P = [[0,360,360,0],
        [0,0,480,480],
        [1,1,1,1]
        ]
    Pdash = np.matmul(h,P)

    tr = [[1,0,500],[0,1,150],[0,0,1]]
    hnew = np.matmul(tr,h)
    # matchesMask = mask.ravel().tolist()
    # Use homography
    height, width= imb.shape
    
    im1Reg = cv2.warpPerspective(imb, hnew, (2400, 1800))

    return im1Reg, h
 

def FinalCall(A,B,min_limit):
    imDownA,imHistA,imBWA,imLPFA = prep(A,w,h,clippinglimit,downsample_level,gaussian_ksize)
    imDownB,imHistB,imBWB,imLPFB = prep(B,w,h,clippinglimit,downsample_level,gaussian_ksize)
    
    ima_kp,ima_desc,keyp_imgA,imb_kp,imb_desc,keyp_imgB = SIFT(imLPFA,imLPFB)

    matches,good,match_img,goodT = Matcher(imBWA,imBWB,ima_kp,imb_kp,ima_desc,imb_desc,min_limit)

    RegImage, PMat = AlignImages(imBWA,imBWB,ima_kp,imb_kp,goodT)
    return match_img,RegImage,PMat



imgA,imgB = inp(img1,img2)

finalimg,alignedImage,PMatrix=FinalCall(imgA,imgB,min_limit)

""" cv2.imshow('DownA',imDownA)
cv2.imshow('HistA',imHistA)
cv2.imshow('BWA',imBWA)
cv2.imshow('LPFA',imLPFA)
cv2.imshow('FA',imFinalA) """
print(PMatrix)
cv2.imshow('Final',finalimg )
cv2.imwrite(Name_Final,alignedImage)
cv2.imshow('Aligned Image',alignedImage)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
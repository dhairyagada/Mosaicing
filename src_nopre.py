import cv2
import numpy as np

img1='InputImages/13.jpeg'
img2='InputImages/14.jpeg'
Name_Final = 'OutputImages/Op2AlignedImg.jpeg'
StitchedImage = 'OutputImages/Op7StitchedNopre.jpeg'
w=360
h=480

min_limit=0.6

def inp(A,B):
    img1=cv2.imread(A)
    img2=cv2.imread(B)
    return img1,img2

def Resize_BW(img,w,h):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, (w,h))
    return img

def SIFT(img):
    sift=cv2.xfeatures2d.SIFT_create()
    kp,desc = sift.detectAndCompute(img,None)
    img_keyp =cv2.drawKeypoints(img,kp,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp,desc,img_keyp

def Matcher(ima,imb,ima_kp,imb_kp,ima_desc,imb_desc,min_limit):
    bf = cv2.BFMatcher(cv2.NORM_L2)

    matches = bf.knnMatch(ima_desc, imb_desc, k=2)
    good = []
    goodT= []
    for m,n in matches:
        if m.distance < min_limit*n.distance:
            good.append([m])
            goodT.append(m)

    match_img = cv2.drawMatchesKnn(ima,ima_kp,imb,imb_kp,good,None,flags=2)

    return matches,good,match_img,goodT
def AlignImages(ima,imb,kp1,kp2,good):
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    points1 = np.float32([ (kp1[m.queryIdx]).pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ (kp2[m.trainIdx]).pt for m in good ]).reshape(-1,1,2)

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    P = [[0,480,480,0],[0,0,360,360],[1,1,1,1]]
    Pdash = np.matmul(h,P)
    Pdash [0:] = Pdash[0:]/Pdash[2:]
    Pdash [1:] = Pdash[1:]/Pdash[2:]
    print(Pdash)
    minx = np.amin(Pdash[0:])
    maxx = np.amax(Pdash[0:])
    miny = np.amin(Pdash[1:])
    maxy = np.amax(Pdash[1:])
    width = maxx-minx
    height = maxy-miny
    txyz = np.dot(h, np.array([imb.shape[1], imb.shape[0], 1]))
    txyz = txyz/txyz[-1]
    dsize = (int(txyz[0])+ima.shape[1], int(txyz[1])+ima.shape[0])
    Tdash = [[1,0,-minx],[0,1,+miny],[0,0,1]]
    #hnew = np.matmul(tr,h)
    hdas = np.matmul(Tdash,h)
    #height, width= imb.shape
    
    im1Reg = cv2.warpPerspective(imb, h, dsize)
    #im1Reg=  cv2.resize(im1Reg, (w,h))
    return im1Reg, h
def FinalCall(A,B,min_limit):
    imBWA = Resize_BW(A,w,h)
    imBWB = Resize_BW(B,w,h)
    
    ima_kp,ima_desc,keyp_imgA = SIFT(imBWA)
    imb_kp,imb_desc,keyp_imgB = SIFT(imBWB)

    matches,good,match_img,goodT = Matcher(imBWA,imBWB,ima_kp,imb_kp,ima_desc,imb_desc,min_limit)
    RegImage, PMat = AlignImages(imBWA,imBWB,ima_kp,imb_kp,goodT)
    #final2 = mix_match(imBWA,RegImage)
    RegImage[0:imBWA.shape[0], 0:imBWA.shape[1]] = imBWA
    final2 =RegImage
    return match_img,RegImage,PMat,final2

imgA,imgB = inp(img1,img2)

finalimg,alignedImage,PMatrix,final2=FinalCall(imgA,imgB,min_limit)

cv2.imshow('Final',finalimg )
#cv2.imwrite(Name_Final,finalimg)
cv2.imshow('Final2',final2)
cv2.imwrite(StitchedImage,final2)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
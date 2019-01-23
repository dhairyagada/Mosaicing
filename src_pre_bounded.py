import cv2
import numpy as np

img1='InputImages/house1.jpeg'
img2='InputImages/house2.jpeg'
Name_Final = 'OutputImages/Op2AlignedImg.jpeg'
StitchedImage = 'OutputImages/OpHouseStitched12.jpeg'
downsample_level=1
clippinglimit=2
w=480
h=360
gaussian_ksize=3
min_limit=0.6

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
    #cv2.rectangle(ima,(x_l,y_l),(xw_l,yw_l),(0,255,0),2)
    #cv2.rectangle(imb,(x_r,y_r),(xw_r,yw_r),(0,255,0),2)
    match_img = cv2.drawMatchesKnn(ima,ima_kp,imb,imb_kp,good,None,flags=2)

    return matches,good,match_img,goodT

def AlignImages(ima,imb,kp1,kp2,good):
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    points1 = np.float32([ (kp1[m.queryIdx]).pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ (kp2[m.trainIdx]).pt for m in good ]).reshape(-1,1,2)

    # Find homography
    Hom, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    txyz = np.dot(Hom, np.array([imb.shape[1], imb.shape[0], 1]))
    txyz = txyz/txyz[-1]
    dsize = (int(txyz[0])+ima.shape[1], int(txyz[1])+ima.shape[0])

    im1Reg = cv2.warpPerspective(imb, Hom, dsize)
    #im1Reg=  cv2.resize(im1Reg, (w,h))
    return im1Reg, Hom
 
def mix_match(leftImage, warpedImage):
    i1y, i1x = leftImage.shape[:2]
    i2y, i2x = warpedImage.shape[:2]
    #print leftImage[-1,-1]

    #t = time.time()
    black_l = np.where(leftImage == np.array([0,0,0]))
    black_wi = np.where(warpedImage == np.array([0,0,0]))
    #print time.time() - t
    #print black_l[-1]

    for i in range(0, i1x):
        for j in range(0, i1y):
            try:
                if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                    # print "BLACK"
                    # instead of just putting it with black, 
                    # take average of all nearby values and avg it.
                    warpedImage[j,i] = [0, 0, 0]
                else:
                    if(np.array_equal(warpedImage[j,i],[0,0,0])):
                        # print "PIXEL"
                        warpedImage[j,i] = leftImage[j,i]
                    else:
                        if not np.array_equal(leftImage[j,i], [0,0,0]):
                            bw, gw, rw = warpedImage[j,i]
                            bl,gl,rl = leftImage[j,i]
                            # b = (bl+bw)/2
                            # g = (gl+gw)/2
                            # r = (rl+rw)/2
                            warpedImage[j, i] = [bl,gl,rl]
            except:
                pass
    # cv2.imshow("waRPED mix", warpedImage)
    # cv2.waitKey()
    return warpedImage
def FinalCall(A,B,min_limit):
    imDownA,imHistA,imBWA,imLPFA = prep(A,w,h,clippinglimit,downsample_level,gaussian_ksize)
    imDownB,imHistB,imBWB,imLPFB = prep(B,w,h,clippinglimit,downsample_level,gaussian_ksize)
    
    ima_kp,ima_desc,keyp_imgA,imb_kp,imb_desc,keyp_imgB = SIFT(imLPFA,imLPFB)

    matches,good,match_img,goodT = Matcher(imBWA,imBWB,ima_kp,imb_kp,ima_desc,imb_desc,min_limit)

    RegImage, PMat = AlignImages(imBWA,imBWB,ima_kp,imb_kp,goodT)
    #final2 = mix_match(imBWA,RegImage)
    final2 = RegImage.copy()
    final2[0:imBWA.shape[0], 0:imBWA.shape[1]] = imBWA
    #final2 =RegImage
    return match_img,RegImage,PMat,final2



imgA,imgB = inp(img1,img2)

finalimg,alignedImage,PMatrix,final2=FinalCall(imgA,imgB,min_limit)

""" cv2.imshow('DownA',imDownA)
cv2.imshow('HistA',imHistA)
cv2.imshow('BWA',imBWA)
cv2.imshow('LPFA',imLPFA)
cv2.imshow('FA',imFinalA) """
print(PMatrix)
cv2.imshow('Final',finalimg )
#cv2.imwrite(Name_Final,alignedImage)
cv2.imshow('Aligned Image',alignedImage)
cv2.imshow('Final2',final2)
cv2.imwrite(StitchedImage,final2)
#cv2.imshow('temp',temp)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
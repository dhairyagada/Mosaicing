import cv2
from matplotlib import pyplot as plt
import numpy as np

### Code For SIFT

def SIFT(imgl,imgr):

    sift=cv2.xfeatures2d.SIFT_create()

    ### Computing the Keypoints and Descriptors
    #For Left Image
    kpl,descl = sift.detectAndCompute(imgl,None)
    #For Right Image
    kpr,descr = sift.detectAndCompute(imgr,None)
    

    ### Drawing Keypoints on the Left and Right Image
    kpImageLeft =cv2.drawKeypoints(imgl,kpl,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpImageRight =cv2.drawKeypoints(imgr,kpr,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return kpl,kpr,descl,descr,kpImageLeft,kpImageRight

### Code For Matching Keypoints

def Matcher(imgl,imgr,iml_kp,imr_kp,iml_desc,imr_desc,min_limit):
    
    bf = cv2.BFMatcher(cv2.NORM_L2)

    ### Matcher Requires the Descriptors of each Keypoint to determine Matches
    matches = bf.knnMatch(iml_desc, imr_desc, k=2)

    good= []

    ### Separating the good matches that satisfy the minimum distance criteria
    for m,n in matches:
        if m.distance < min_limit*n.distance:
            good.append(m)

    ### Drawing The Matches 
    match_img = cv2.drawMatchesKnn(imgl,iml_kp,imgr,imr_kp,[good],None,flags=2)

    return good,match_img

def calcHom(imageLeft,imageRight,kpl,kpr,good):
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    points1 = np.float32([ (kpl[m.queryIdx]).pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ (kpr[m.trainIdx]).pt for m in good ]).reshape(-1,1,2)

    # Find homography
    
    Hom, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    return Hom,mask



def WarpAndStitch(imageLeft,imageRight,Hom):

    """ txyz = np.dot(Hom, np.array([imageRight.shape[1], imageRight.shape[0], 1]))
    txyz = txyz/txyz[-1]
    dsize = (int(txyz[0])+imageLeft.shape[1], int(txyz[1])+imageLeft.shape[0]) """
    dsize = (500,500)
    Warped_Img = cv2.warpPerspective(imageRight, Hom, dsize)
    
    FinalImage = Warped_Img.copy()
    FinalImage[0:imageLeft.shape[0], 0:imageLeft.shape[1]] = imageLeft
    #final2 =RegImage
    return Warped_Img,FinalImage


def descrfunc(imgA,imgB):
    minlimit = 0.6               ## Minimum Distance Limit
    A = cv2.resize(imgA,(200,200))
    B = cv2.resize(imgB,(200,200))
    Hom =  np.zeros((3,3))
    kpl,kpr,descl,descr,kpImageLeft,kpImageRight = SIFT(A,B)
    ##Hom,mask = calcHom(A,B,kpl,kpr,good)
    
    """ scale = 0.5
    Hom[0,2] = Hom[0,2]/scale
    Hom[1,2] = Hom[1,2]/scale
    Hom[2,0] = scale*Hom[2,0]
    Hom[2,1] = scale*Hom[2,1]  """

    #Warped_Img,Final = WarpAndStitch(imgA,imgB,Hom)

    return kpl,kpr,descl,descr,kpImageLeft,kpImageRight
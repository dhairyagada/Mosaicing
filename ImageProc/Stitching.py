from myconfig import *
import cv2
import numpy as np

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
    dsize = (600,600)
    Warped_Img = cv2.warpPerspective(imageRight, Hom, dsize)
    
    FinalImage = Warped_Img.copy()
    FinalImage[0:imageLeft.shape[0], 0:imageLeft.shape[1]] = imageLeft
    #final2 =RegImage
    return Warped_Img,FinalImage

def ImageStitcher(imageLeft,imageRight,kpl,kpr,good):

    Hom,mask = calcHom(imageLeft,imageRight,kpl,kpr,good)
    Warped_Img,FinalImage = WarpAndStitch(imageLeft,imageRight,Hom)

    return Hom,Warped_Img,FinalImage


def mix_and_match(leftImage, rightImage):
   
    MixImage = leftImage.copy()
    """ 
    iLimit = leftImage.shape[0]
    jLimit = leftImage.shape[1]

    for i in range(0,iLimit-1):
        for j in range(0, jLimit-1):
            
            if (leftImage[i,j] == [0,0,0]).all() and (rightImage[i,j] != [0,0,0]).all():
                MixImage[i,j] = rightImage[i,j]
            elif (leftImage[i,j] != [0,0,0]).all() and (rightImage[i,j] == [0,0,0]).all():
                MixImage[i,j] = leftImage[i,j]
            else:
                MixImage[i,j] = leftImage[i,j] """
    MixImage = cv2.addWeighted(leftImage,0.7,rightImage,0.3,0)
    return MixImage
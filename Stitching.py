from myconfig import *
import cv2
import numpy as np

def ImageStitcher(imageLeft,imageRight,kpl,kpr,good):
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    points1 = np.float32([ (kpl[m.queryIdx]).pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ (kpr[m.trainIdx]).pt for m in good ]).reshape(-1,1,2)

    # Find homography
    Hom, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    txyz = np.dot(Hom, np.array([imageRight.shape[1], imageRight.shape[0], 1]))
    txyz = txyz/txyz[-1]
    dsize = (int(txyz[0])+imageLeft.shape[1], int(txyz[1])+imageLeft.shape[0])

    Warped_Img = cv2.warpPerspective(imageRight, Hom, dsize)
    
    FinalImage = Warped_Img.copy()
    FinalImage[0:imageLeft.shape[0], 0:imageLeft.shape[1]] = imageLeft
    #final2 =RegImage
    return Hom,Warped_Img,FinalImage

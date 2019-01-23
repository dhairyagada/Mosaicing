import cv2
import numpy as np 
from myconfig import *
from PreProcessing import *
from Matching import *
from Stitching import *

def inputImages(A,B):
    imgA = cv2.imread(A)
    imgB = cv2.imread(B)
    return imgA,imgB

imgA,imgB = inputImages(img1,img2)

imgAp = PreProcess(imgA)
imgBp = PreProcess(imgB)

kpl,kpr,descl,descr,kpImageLeft,kpImageRight,goodT,match_img = KeypointMatcher(imgAp,imgBp) 

imgACol = cv2.resize(imgA,(w,h))
imgBcol = cv2.resize(imgB,(w,h))

P_Matrix,WarpedImage,StitchedImage = ImageStitcher(imgACol,imgBcol,kpl,kpr,goodT)

cv2.imshow("Image 1",imgAp)
cv2.imshow("Image 2",imgBp)

cv2.imshow("Left Image Keypoints",  kpImageLeft)
cv2.imshow("Right Image Keypoints", kpImageLeft)

cv2.imshow("Keypoint Matches",match_img)

cv2.imshow("Warped Image",WarpedImage)
cv2.imshow("Stitched Image",StitchedImage)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
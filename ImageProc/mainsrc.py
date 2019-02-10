import cv2
import numpy as np 
from myconfig import *
from PreProcessing import *
from Matching import *
from Stitching import *
import time

start_time = time.time()

def inputImages(A,B,C):
    imgA = cv2.imread(A)
    imgB = cv2.imread(B)
    imgC = cv2.imread(C)
    return imgA,imgB,imgC

imgA,imgB,imgC = inputImages(img1,img2,img3)

imgAp = PreProcess(imgA)
imgBp = PreProcess(imgB)
imgCp = PreProcess(imgC)

cv2.imshow("Image 1",imgAp)
cv2.imshow("Image 2",imgBp)
cv2.imshow("Image 3",imgCp)

imgACol = cv2.resize(imgA,(w,h))
imgBcol = cv2.resize(imgB,(w,h))
imgCCol = cv2.resize(imgC,(w,h))

## Homography for Image 1 and 2
kpl1,kpr1,descl1,descr1,kpImageLeft1,kpImageRight1,goodT1,match_img1 = KeypointMatcher(imgAp,imgBp)
P_Matrix1,WarpedImage1,StitchedImage1 = ImageStitcher(imgACol,imgBcol,kpl1,kpr1,goodT1)
print(P_Matrix1)
print("abc")
Stitch1P = PreProcess(StitchedImage1)
Stitch1Col = cv2.resize(StitchedImage1,(w,h))

## Homography for Image 2 and 3
kpl2,kpr2,descl2,descr2,kpImageLeft2,kpImageRight2,goodT2,match_img2 = KeypointMatcher(imgBp,imgCp)
P_Matrix2,WarpedImage2,StitchedImage2 = ImageStitcher(imgBcol,imgCCol,kpl2,kpr2,goodT2)

## Homography for 
P_Matrix3 = np.matmul(P_Matrix1,P_Matrix2)

WarpedImage3,StitchedImage3 = WarpAndStitch(StitchedImage1,imgCCol,P_Matrix3)

FinalStitch = mix_and_match(StitchedImage1,WarpedImage3)

end_time = time.time()
print("----%s seconds"%(end_time-start_time))
cv2.imshow("Match 1",match_img1)
cv2.imshow("Match 2",match_img2)
cv2.imshow("Stitched 1 and 2",  StitchedImage1)
cv2.imshow("Stitched 2 and 3",  StitchedImage2)
#cv2.imshow("Warped 3 ",WarpedImage3)
#cv2.imshow("Warped Image",WarpedImage)
cv2.imshow("Stitched Image",FinalStitch)
cv2.imwrite('OutputImages/homeimagestitchnopre.jpeg',FinalStitch)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
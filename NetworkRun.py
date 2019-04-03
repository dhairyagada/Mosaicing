import cv2
import numpy as np
from ImageProc.Stitching import *
from HNet import *
import time
start_time = time.time()

""" ipimg1 = input('Enter Image 1 path : ')
ipimg2 = input('Enter Image 2 path : ')
ipimg3 = input('Enter Image 3 path : ') """

imgA = cv2.imread(img1)
imgA = cv2.resize(imgA,(w,h))
imgB = cv2.imread(img2)
imgB = cv2.resize(imgB,(w,h))
imgC = cv2.imread(img3)
imgC = cv2.resize(imgC,(w,h))

print("Stitching 1 and 2")
KpX1,KpY1,matches1,Hom1,Warped_Img1,Stitch1 = NetRunHom(imgA,imgB)
print("Stitching 2 and 3")
KpX2,KpY2,matches2,Hom2,Warped_Img2,Stitch2 = NetRunHom(imgB,imgC)
print("Stitching all together")
Hom3 = np.matmul(Hom1,Hom2)

Warped_Img3,Stitch3 = WarpAndStitch(Stitch1,imgC,Hom3)
FinalStitch = mix_and_match(Stitch1,Warped_Img3)

end_time = time.time()
print("----%s seconds"%(end_time-start_time))

cv2.imwrite('match1-net.jpg',matches1)
cv2.imwrite('match2-net.jpg',matches2)
# cv2.imshow('KpX1',KpX1)
# cv2.imshow('KpY1',KpY1)
cv2.imshow('match1',cv2.imread('./match1-net.jpg'))
cv2.imshow('match2',cv2.imread('./match2-net.jpg'))
# cv2.imshow('Warped1',Warped_Img1)
cv2.imshow('Image1',imgA)
cv2.imshow('Image2',imgB)
cv2.imshow('Image3',imgC)
cv2.imshow('Stitch1',Stitch1)
cv2.imshow('Stitch2',Stitch2)
cv2.imshow('FinalSTitch',FinalStitch)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

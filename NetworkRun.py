import cv2
import numpy as np
from ImageProc.Stitching import *
from HNet import *
from ImageProc.cylinderwarp import *
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
imgD = cv2.imread(img4)
imgD = cv2.resize(imgD,(w,h))
imgE = cv2.imread(img5)
imgE = cv2.resize(imgE,(w,h))
""" imgA = cylindricalWarp(imgA)
imgB = cylindricalWarp(imgB)
imgC = cylindricalWarp(imgC) """

print("Stitching 1 and 2")
KpX1,KpY1,matches1,Hom1,Warped_Img1,Stitch1 = NetRunHom(imgA,imgB)
print("Stitching 2 and 3")
KpX2,KpY2,matches2,Hom2,Warped_Img2,Stitch2 = NetRunHom(imgB,imgC)
print("Stitching 3 and 4")
KpX3,KpY3,matches3,Hom3,Warped_Img3,Stitch3 = NetRunHom(imgC,imgD)
print("Stitching 4 and 5")
KpX4,KpY4,matches4,Hom4,Warped_Img4,Stitch4 = NetRunHom(imgD,imgE)
print("Stitching all together")

Hom13 = np.matmul(Hom1,Hom2)
Hom14 = np.matmul(Hom13,Hom3)
Hom15 = np.matmul(Hom14,Hom4)

print("a")
Warped_Img3,St = WarpAndStitch(Stitch1,imgC,Hom13)
Stitch3 = mix_and_match(Stitch1,Warped_Img3)
print("a")
Warped_Img4,St = WarpAndStitch(Stitch1,imgD,Hom14)
Stitch4 = mix_and_match(Stitch3,Warped_Img4)
print("a")
Warped_Img5,St = WarpAndStitch(Stitch1,imgE,Hom15)
Stitch5 = mix_and_match(Stitch4,Warped_Img5)



end_time = time.time()
print("----%s seconds"%(end_time-start_time))

cv2.imwrite('match1-net.jpg',matches1)
cv2.imwrite('match2-net.jpg',matches2)
# cv2.imshow('KpX1',KpX1)
# cv2.imshow('KpY1',KpY1)
# cv2.imshow('match1',cv2.imread('./match1-net.jpg'))
# cv2.imshow('match2',cv2.imread('./match2-net.jpg'))
# cv2.imshow('Warped1',Warped_Img1)
# cv2.imshow('Image1',imgA)
# cv2.imshow('Image2',imgB)
# cv2.imshow('Image3',imgC)
# cv2.imshow('Image4',imgD)
# cv2.imshow('Image5',imgE)
cv2.imshow('Stitch1',Stitch1)
cv2.imshow('Stitch3',Stitch3)
cv2.imshow('Stitch4',Stitch4)
cv2.imshow('Stitch5',Stitch5)
#cv2.imshow('FinalSTitch',FinalStitch)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

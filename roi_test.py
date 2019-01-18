import cv2
import numpy as np

img1='InputImages/7.jpeg'
img2='InputImages/8.jpeg'
w=360
h=480


def inp(A,B):
    img1=cv2.imread(A)
    img2=cv2.imread(B)
    return img1,img2

def Resize_BW(img,w,h):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, (w,h))
    return img


imgA,imgB = inp(img1,img2)

A = Resize_BW(imgA,w,h)
B = Resize_BW(imgB,w,h)

cv2.rectangle(A,(160,0),(360,480),(0,255,0),3)

cv2.imshow('Final',A[0:480,160:360])

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
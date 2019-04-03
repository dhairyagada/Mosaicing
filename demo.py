from __future__ import print_function
import Registration
import matplotlib.pyplot as plt
from util import *
import cv2
import numpy as np
from time import sleep
from ImageProc.cylinderwarp import *
from myconfig import *
# designate image path here
#plt.ion()

def show(imgX):
    plt.plot()
    plt.imshow(cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB))
    plt.show()
    return

climit = 2
def iscollinear(x1, y1, x2, y2, x3, y3): 
      
    """ Calculation the area of   
        triangle. We have skipped  
        multiplication with 0.5 to 
        avoid floating point computations """
    a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    
    return a

def Hist_Eq(imgT,climit):
    
    image_lab = cv2.cvtColor(imgT,cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    merged_channels = cv2.merge((cl, a_channel, b_channel))
    G = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return G

#IX_path = './ImageProc/InputImages/clg1.jpg'
#IY_path = './ImageProc/InputImages/clg2.jpg'

IX = cv2.imread(img1)
IY = cv2.imread(img2)

IX = cv2.resize(IX,(w,h))
IY = cv2.resize(IY,(w,h))

matches = np.zeros((IX.shape[0],IX.shape[1]*2,3))

matches[:IX.shape[0],:IX.shape[1]] = IX
matches[:IX.shape[0],IX.shape[1]:IX.shape[1]*2] = IY 
#show(matches)
cv2.imwrite('matches-net.jpeg',matches)
IX1 = Hist_Eq(IX,climit)
IY1 = Hist_Eq(IY,climit)

#initialize
reg = Registration.CNN()
#register
X, Y, Z = reg.register(IX1, IY1)

Xdash = X.copy()
Ydash = Y.copy()
Zdash = Z.copy()
for j in range(len(X)):
    Xdash[j,1] = X[j,0]
    Xdash[j,0] = X[j,1]

    Ydash[j,1] = Y[j,0]
    Ydash[j,0] = Y[j,1]

    Zdash[j,1] = Z[j,0]
    Zdash[j,0] = Z[j,1]

for i in range(len(X)):
    cv2.circle(IX,(int(X[i,1]),int(X[i,0])),8, (0,0,255), 6)
    cv2.circle(IY,(int(Y[i,1]),int(Y[i,0])),8, (0,0,255), 6)

    cv2.circle(matches,(int(X[i,1]),int(X[i,0])),8, (0,0,255), 6)
    cv2.circle(matches,(int(Y[i,1])+IX.shape[1],int(Y[i,0])),8, (0,0,255), 6)

    cv2.line(matches,(int(X[i,1]),int(X[i,0])),(int(Y[i,1])+IX.shape[1],int(Y[i,0])),(0,255,255),5)
    """ plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(IY, cv2.COLOR_BGR2RGB))
    plt.show() """
cv2.imshow('matches',matches)
cv2.imwrite('matches-net1.jpeg',matches)
points1 = np.float32(Zdash).reshape(-1,1,2)
points2 = np.float32(Ydash).reshape(-1,1,2)

c = 2
Ax = Xdash[0]
Ay = Ydash[0]
Az = Zdash[0]

Bx = Xdash[1]
By = Ydash[1]
Bz = Zdash[1]
 
Cx = Xdash[c]
Cy = Ydash[c]
Cz = Zdash[c]

c = c+1
Dx = Xdash[c]
Dy = Ydash[c]
Dz = Zdash[c]

while(c<120):
    flag = 0 
    #print(iscollinear(Ax[0],Ax[1],Bx[0],Bx[1],Cx[0],Cx[1]))
    if (iscollinear(Ax[0],Ax[1],Bx[0],Bx[1],Cx[0],Cx[1])<600):
        c = c + 1
        Cx = Xdash[c]
        Cy = Ydash[c]
        Cz = Zdash[c]

    else:
        flag = flag + 1
    if(iscollinear(Ax[0],Ax[1],Bx[0],Bx[1],Dx[0],Dx[1])<600):
        c = c+1
        Dx =Xdash[c]
        Dy = Ydash[c]
        Dz = Zdash[c]

    else:
        flag = flag + 1
    if(iscollinear(Bx[0],Bx[1],Cx[0],Cx[1],Dx[0],Dx[1])<600):
        c = c + 1
        Bx = Xdash[c]
        By = Ydash[c]
        Bz = Zdash[c]

    else:
        flag = flag + 1
    if(iscollinear(Cx[0],Cx[1],Dx[0],Dx[1],Ax[0],Ax[1])<600):
        c = c + 1
        Ax = Xdash[c]
        Ay = Ydash[c]
        Az = Zdash[c]

    else:
        flag = flag + 1
    
    if flag == 4:
        break
    
cv2.circle(IX,(int(Ax[0]),int(Ax[1])),3, (0,0,255), -1)
cv2.circle(IX,(int(Bx[0]),int(Bx[1])),3, (0,0,255), -1)
cv2.circle(IX,(int(Cx[0]),int(Cx[1])),3, (0,0,255), -1)
cv2.circle(IX,(int(Dx[0]),int(Dx[1])),3, (0,0,255), -1)
""" 
c=2
Cy = Ydash[c]
c = c+1
Dy = Ydash[c]

while(c<120):
    flag = 0 
    #print(iscollinear(Ax[0],Ax[1],Bx[0],Bx[1],Cx[0],Cx[1]))
    if (iscollinear(Ay[0],Ay[1],By[0],By[1],Cy[0],Cy[1])<100):
        c = c + 1
        Cy = Ydash[c]
    else:
        flag = flag + 1
    if(iscollinear(Ay[0],Ay[1],By[0],By[1],Dy[0],Dy[1])<100):
        c = c+1
        Dy =Ydash[c]
    else:
        flag = flag + 1
    if(iscollinear(By[0],By[1],Cy[0],Cy[1],Dy[0],Dy[1])<100):
        c = c + 1
        By = Ydash[c]
    else:
        flag = flag + 1
    if(iscollinear(Cy[0],Cy[1],Dy[0],Dy[1],Ay[0],Ay[1])<100):
        c = c + 1
        Ay = Ydash[c]
    else:
        flag = flag + 1
    
    if flag == 4:
        break
"""
cv2.circle(IY,(int(Ay[0]),int(Ay[1])),3, (0,0,255), -1)
cv2.circle(IY,(int(By[0]),int(By[1])),3, (0,0,255), -1)
cv2.circle(IY,(int(Cy[0]),int(Cy[1])),3, (0,0,255), -1)
cv2.circle(IY,(int(Dy[0]),int(Dy[1])),3, (0,0,255), -1)

#points1 = np.float32([Az,Bz,Cz,Dz])
#points2 = np.float32([Ay,By,Cy,Dy])

#Hom = cv2.getPerspectiveTransform(points2,points1)

Hom, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
Warped_Img = cv2.warpPerspective(IY, Hom, (600,600))

FinalImage = Warped_Img.copy()
FinalImage[0:IX.shape[0], 0:IX.shape[1]] = IX
#generate regsitered image using TPS
#registered = tps_warp(Y, Z, IY, (600,600,3))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(IY, cv2.COLOR_BGR2RGB))
plt.show()

plt.plot()
plt.imshow(cv2.cvtColor(Warped_Img, cv2.COLOR_BGR2RGB))
plt.show()

plt.plot()
plt.imshow(cv2.cvtColor(FinalImage, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imshow('matches',matches)

cv2.imwrite('final.jpeg',FinalImage)

cv2.imwrite('Keypoint-Net23.jpeg',IX)
cv2.imwrite('Keypoint-Net3.jpeg',IY)



print("Hello")

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
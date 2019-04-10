import cv2
import numpy as np
from ImageProc.Stitching import *
from HNet import *
from ImageProc.cylinderwarp import *
import time
from glob import glob
from matplotlib import pyplot as plt

start_time = time.time()
#rawdatapath = "./ImageProc/InputImages/a/*.jpg"

""" ipimg1 = input('Enter Image 1 path : ')
ipimg2 = input('Enter Image 2 path : ')
ipimg3 = input('Enter Image 3 path : ') """
loc_list = sorted(glob(rawdatapath))
len_list = len(loc_list)
# index = randint(0,numrawimages-1)
# colimgloc = loc_list[index]
def readimage(img):
    x = cv2.imread(img)
    x = cv2.resize(x,(w,h))
    x = cylindricalWarp(x)
    return x
def show(im):
    plt.plot()
    plt.imshow(im)
    plt.show()
    return

""" for j in range(len_list):
    show(cv2.imread(loc_list[j])) """
interval = 20
A = readimage('./ImageProc/InputImages/b/0.jpg')
B = readimage('./ImageProc/InputImages/b/%d.jpg' %interval)
KpX,KpY,matches,Hom,Warped_Img,StitchBase = NetRunHom(A,B)

HomAcc = Hom.copy()
StitchFin = StitchBase.copy()
cv2.imwrite("Stitchfin.jpg",StitchFin)
#show(StitchFin)
for i in range (interval,len_list-interval-1,interval):
    print(i)
    left = readimage('./ImageProc/InputImages/b/%d.jpg' %i)
    right = readimage('./ImageProc/InputImages/b/%d.jpg' %(i+interval))

    KpXT,KpYT,matchesT,HomT,Warped_ImgT,StitchT = NetRunHom(left,right)
    HomAcc = np.matmul(HomAcc,HomT)

    WarpedImgT, StitchT = WarpAndStitch(A,right,HomAcc)
    StitchFin = mix_and_match(StitchFin,WarpedImgT)
    cv2.imwrite("Stitchfin.jpg",StitchFin)
    """ show(left)
    show(right)
    show(StitchFin) """

cv2.imshow('Stitch',StitchFin)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

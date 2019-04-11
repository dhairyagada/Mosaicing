import cv2
import numpy as np 
from myconfig import *
from PreProcessing import *
from Matching import *
from Stitching import *
import time
from cylinderwarp import *
from matplotlib import pyplot as plt
from glob import glob

start_time = time.time()

loc_list = sorted(glob("./ImageProc/InputImages/b/*.jpg"))
len_list = len(loc_list)

def show(im):
    plt.plot()
    plt.imshow(im)
    plt.show()
    return
def readimage(img):
    x = cv2.imread(img)
    x = cv2.resize(x,(w,h))
    x = cylindricalWarp(x)
    x_col = x.copy()
    x = Hist_Eq(x,clippinglimit)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return x_col,x
interval = 1
imgACol,imgAp = readimage('./ImageProc/InputImages/b/0.jpg')
imgBCol,imgBp = readimage('./ImageProc/InputImages/b/%d.jpg' %interval)
show(imgACol)
show(imgAp)
kpl1,kpr1,descl1,descr1,kpImageLeft1,kpImageRight1,goodT1,match_img1 = KeypointMatcher(imgAp,imgBp)
P_Matrix1,WarpedImage1,StitchedImage1 = ImageStitcher((imgACol),(imgBCol),kpl1,kpr1,goodT1)

HomAcc  = P_Matrix1.copy()
StitchFin = StitchedImage1.copy()

for i in range (interval,len_list-interval,interval):
    print(i)
    leftCol,left = readimage('./ImageProc/InputImages/b/%d.jpg' %i)
    rightCol,right = readimage('./ImageProc/InputImages/b/%d.jpg' %(i+interval))

    kplT,kprT,desclT,descrT,kpImageLeftT,kpImageRightT,goodT,match_imgT = KeypointMatcher(left,right)
    HomT,WarpedImageT,StitchedT = ImageStitcher((leftCol),(rightCol),kplT,kprT,goodT)
    HomAcc = np.matmul(HomAcc,HomT)

    WarpedImgT, StitchT = WarpAndStitch(imgACol,rightCol,HomAcc)
    StitchFin = mix_and_match(StitchFin,WarpedImgT)
    #show(StitchFin)
    cv2.imwrite("Stitchfin.jpg",StitchFin)

#show(StitchFin)
cv2.imshow("StitchFin",StitchFin)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

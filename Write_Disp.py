import cv2
import numpy as np

def show(img_array):
    l=len(img_array)
    i=1
    cv2.imshow(i,img_array[1])
    cv2.imshow('b',img_array[0])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

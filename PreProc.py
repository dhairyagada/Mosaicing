import cv2
import numpy as np

def Hist_Eq(img,climit):
    image_lab = cv2.imread(img,cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    merged_channels = cv2.merge((cl, a_channel, b_channel))
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image

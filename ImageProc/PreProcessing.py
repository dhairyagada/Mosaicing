from myconfig import *
import cv2
import numpy as np
def Downsample(imgT,downlevel,wid,ht):
    
    xrange=range
    G = imgT
    for i in xrange(downlevel):
        G = cv2.pyrDown(G)
    G=cv2.resize(G,(w,h))
    return G

def Hist_Eq(imgT,climit):
    
    image_lab = cv2.cvtColor(imgT,cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    merged_channels = cv2.merge((cl, a_channel, b_channel))
    G = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return G

def Resize_BW(imgT,wid,ht):
    
    G=cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
    G=cv2.resize(G, (wid,ht))
    return G

def GaussianLPF(imgT,k_size):
    
    G=cv2.GaussianBlur(imgT,(k_size,k_size),0)
    return G

def PreProcess(imgT):
    if ifProc == 0:
        G = Resize_BW(imgT,w,h)
        return G
    else:
        G1 = Downsample(imgT,downsample_level,w,h)
        G2 = Hist_Eq(G1,clippinglimit)
        G3 = Resize_BW(G2,w,h)
        G4 = GaussianLPF(G3,gaussian_ksize)
        return G4
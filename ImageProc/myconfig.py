# Configurations

## Input Images
img1='./ImageProc/InputImages/2c.jpg'
img2='./ImageProc/InputImages/3c.jpg'
img3='./ImageProc/InputImages/1c.jpg'
## Image ReSizing Parameters
w=360
h=480

## Pre-Processing Parameters

ifProc = 0

# Downsampling
downsample_level=1
# Histogram Equalization
clippinglimit=2
# Gaussian Filter
gaussian_ksize=3

## Feature Matching Limit

min_limit=0.7

## Region of Interest

ROIDisp = 1
ifROI = 0
# Left Image
x_l     = 200
xw_l    = 360

y_l     =  0
yw_l    =  480

# Right Image
x_r     = 0
xw_r    = 160

y_r     = 0
yw_r    = 480


## Cylindrical Warping

focal_len = 3000
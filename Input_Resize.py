import cv2

## Size

def input_images(A):
    image = cv2.imread(A)
    ims1 = cv2.resize(image1, (360,480))
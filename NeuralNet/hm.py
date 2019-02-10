from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, ELU, BatchNormalization, Lambda, merge, MaxPooling2D, Input, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.utils import plot_model
from keras.optimizers import Adam,SGD
from keras.callbacks import Callback, RemoteMonitor
import keras.backend as K

from glob import glob
from matplotlib import pyplot as plt
from numpy.linalg import inv
import cv2
import random
import numpy as np
import time

from Stitching import *
from myconfig import *
print(w)
def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.maximum(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True), K.epsilon()))

def homography_regression_model():
    input_shape=(128, 128, 2)
    input_img = Input(shape=input_shape)
     
    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv1")(input_img)
    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv2")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    
    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv3")(x)
    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv4")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
   
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv5")(x)
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv6")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)
    
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv7")(x)
    x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv8")(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    x = Dropout(0.75, noise_shape=None, seed=None)(x)
    x = Dense(1024, name='FC1')(x)
    out = Dense(8, name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    #plot_model(model, to_file='documentation_images/HomegraphyNet_Regression.png', show_shapes=True)
    
    model.compile(optimizer=Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=euclidean_distance)
    return model

def get_test(path1,path2):
    rho = 32
    patch_size = 128
    height = 480
    width = 360
    
    color_image = cv2.imread(path1)
    color_image = cv2.resize(color_image,(width,height))
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

    warped_image = cv2.imread(path2)
    warped_image = cv2.resize(warped_image,(width,height))
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
    #points
    y = random.randint(rho, height - rho - patch_size)  # row
    x = random.randint(rho,  width - rho - patch_size)  # col
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points_array = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    #four_points_array = np.array(four_points)   
    
    
    # grab image patches
    original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    warped_patch = warped_image[y:y + patch_size, x:x + patch_size]
    # make into dataset
    training_image = np.dstack((original_patch, warped_patch))
    val_image = training_image.reshape((1,128,128,2))
    cv2.imshow("Og",original_patch)
    cv2.imshow("Warped",warped_patch)
    return color_image,val_image,four_points_array

model = homography_regression_model()
#model.summary()
K.clear_session()
model = homography_regression_model()
model.load_weights('./NeuralNet/my_model_weights.h5')

color_image,val_image,four_points_array = get_test("./NeuralNet/home1edit.jpg","./NeuralNet/home2edit.jpg")

labels = model.predict(val_image)
K.clear_session()
labels_ = np.float32(labels.reshape((4,2)))

perturbed_four = np.float32(np.subtract(four_points_array,labels_))
fourpoints = np.float32(four_points_array)
#print(fourpoints)
#print(perturbed_four)
Hdash = cv2.getPerspectiveTransform(fourpoints,perturbed_four)
print('H =',Hdash)
#print(labels)
#print(labels_)
""" plt.imshow(rectangle_image) 
plt.title('original image')  
plt.show()

plt.imshow(warped_image) 
plt.title('warped_image image')
plt.show()
print ('done') """

def inputImages(A,B,C):
    imgA = cv2.imread(A)
    imgB = cv2.imread(B)
    imgC = cv2.imread(C)
    return imgA,imgB,imgC

imgA,imgB,imgC = inputImages(img1,img2,img3)

imgACol = cv2.resize(imgA,(w,h))
imgBcol = cv2.resize(imgB,(w,h))
imgCCol = cv2.resize(imgC,(w,h))

warpedimg,finalimg = WarpAndStitch(imgACol,imgBcol,Hdash)
cv2.imshow("Warped2",warpedimg)
cv2.imshow("Final",finalimg)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
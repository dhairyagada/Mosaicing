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
import pickle

from GetTest import *
from mynetconfig import *
from Stitching import *

def inputImages(A,B):
    imgA = cv2.imread(A)
    imgB = cv2.imread(B)
    return imgA,imgB

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

def train_network():
    train_file = 'train_new.p'
    valid_file = 'valid_new.p'

    with open(train_file, mode='rb') as f:
        train = pickle.load(f)
    with open(valid_file, mode='rb') as f:
        valid = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    print("X_train shape = ", X_train.shape)
    print("y_train shape = ", y_train.shape)

    K.clear_session()
    model = homography_regression_model()
    model.load_weights('NetWeights.h5')
    h = model.fit(x=X_train, y=y_train, verbose=1, batch_size=20, nb_epoch=2, validation_split=0.3)
    model.save_weights('NetWeights.h5')
    K.clear_session()
    model = homography_regression_model()
    model.load_weights('NetWeights.h5')
    y_test=model.predict(X_valid)
    print('Training Done , Weights Saved to NetWeights.h5')
    return


def HPrediction():
    print("Input Test Started")
    pathA = './ImageProc/InputImages/cycle1.jpeg'
    pathB = './ImageProc/InputImages/cycle2.jpeg'

    inputPatches,points = GetInputPatches(pathA,pathB)
    print(inputPatches.shape)

    model = homography_regression_model()
    model.load_weights('NetWeights.h5')
    y_predicted=model.predict(inputPatches)

    K.clear_session()
    ypred = np.float32(y_predicted.reshape((4,2)))

    perturbed_four = np.float32(np.subtract(points,ypred))
    fourpoints = np.float32(points)
    #print(fourpoints)
    #print(perturbed_four)
    Hdash = cv2.getPerspectiveTransform(fourpoints,perturbed_four)
    print('H =',Hdash)

    imgA,imgB = inputImages(pathA,pathB)

    imgACol = cv2.resize(imgA,(w,h))
    imgBcol = cv2.resize(imgB,(w,h))

    warpedimg,finalimg = WarpAndStitch(imgACol,imgBcol,Hdash)
    #finimg = mix_and_match(imgACol,warpedimg)
    cv2.imshow("Warped2",warpedimg)
    cv2.imshow("Final",finalimg)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

    return
train_network()
#HPrediction()


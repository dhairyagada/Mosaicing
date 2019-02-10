import pickle
from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
import os

# hyperparameters
rho = 32
patch_size = 128
height = 240
width = 320
visualize = False
num_examples = 10
outpath = ("./NeuralNet/SampleTestSet/new")

loc_list = glob("./NeuralNet/SampleTestSet/*.jpg")
X = np.zeros((128, 128, 2, num_examples))  # images
Y = np.zeros((4, 2, num_examples))
for i in range(num_examples):
    if i % 2 == 0:
        print("Created ", i, " examples.")
    # select random image from tiny training set
    index = random.randint(0, 9)
    img_file_location = loc_list[index]
    color_image = plt.imread(img_file_location)
    color_image = cv2.resize(color_image, (width, height))
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

    # create random point P within appropriate bounds
    y = random.randint(rho, height - rho - patch_size)  # row?
    x = random.randint(rho, width - rho - patch_size)  # col?
    # define corners of image patch
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # compute H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)
    inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 240))
    warped_image = cv2.warpPerspective(gray_image, H, (320, 240))

    # grab image patches
    original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
    # make into dataset
    training_image = np.dstack((original_patch, warped_patch))
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    X[:, :, :, i] = training_image
    Y[:, :, i] = H_four_points
    save_name_og = os.path.join(outpath,os.path.basename('Og'+str(i))+'.jpg')
    save_name_wp = os.path.join(outpath,os.path.basename('Warped'+str(i))+'.jpg')
    cv2.imwrite(save_name_og,gray_image)
    cv2.imwrite(save_name_wp,warped_image)
print("Saving in pickle format.")
X_train = X[0:int(0.9 * num_examples)]
X_valid = X[int(0.9 * num_examples):]
Y_train = Y[0:int(0.9 * num_examples)]
Y_valid = Y[int(0.9 * num_examples):]
train = {'features': X_train, 'labels': Y_train}
valid = {'features': X_valid, 'labels': Y_valid}
pickle.dump(train, open("train.p", "wb"))
pickle.dump(valid, open("valid.p", "wb"))
print("Done.")
cv2.imshow("Win1",gray_image)
cv2.imshow("Win2",warped_image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
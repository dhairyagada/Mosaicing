import cv2
vidObj = cv2.VideoCapture('./ImageProc/InputImages/ken.mp4') 
# Used as counter variable 
count = 0
# checks whether frames were extracted 
success = 1
while success: 

    # vidObj object calls read 
    # function extract frames 
    success, image = vidObj.read()
    h = image.shape[0]/2
    w = image.shape[1]/2
    M = cv2.getRotationMatrix2D((w,h),180, 1.0)
    image = cv2.warpAffine(image, M, (int(w*2), int(h*2)))
    # Saves the frames with frame-count 
    cv2.imwrite("./ImageProc/InputImages/b/%d.jpg" % count, image) 
    print(count)
    count += 1

print('done')
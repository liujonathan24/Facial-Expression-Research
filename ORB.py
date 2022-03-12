#ORB
import os
import dlib
import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt

for a in ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]:
    faces_folder_path = "./New-CK+/" + a
    for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
        print("Processing file: {}".format(f))
        src = cv2.imread(f, 0)
        src = src[1:129][1:129]
        print(src.shape)
        print('hi')
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(src, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(src, kp)
        print(des.shape)
        print(len(kp))
        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(src, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2), plt.show()
        break
    break
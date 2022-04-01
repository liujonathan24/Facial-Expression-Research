import numpy as np
#ORB
import os
import dlib
import glob
import cv2 as cv
from matplotlib import pyplot as plt



for a in ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]:
    faces_folder_path = "./New-CK+/" + a
    for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
        print("Processing file: {}".format(f))
        img = cv.imread(f, 0)

        # Initiate FAST detector
        star = cv.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        # find the keypoints with STAR
        kp = star.detect(img, None)
        # compute the descriptors with BRIEF
        kp, des = brief.compute(img, kp)
        print(brief.descriptorSize())
        print(des.shape)



        break
    break
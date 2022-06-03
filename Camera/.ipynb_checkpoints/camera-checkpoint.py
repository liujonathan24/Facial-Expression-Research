import os
import dlib
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import scipy
import pandas as pd
import random
import math


import cv2 as cv



def crop_and_vector(src, parts_list):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
    # Initiate ORB detector
    orb = cv2.ORB_create()

    face_area = detector(src, 1)
    shape = predictor(src, face_area[0])
    pts = get_face_points_array(parts_list, shape)

    # Create polygon shaped mask
    mask = np.zeros((src.shape[0], src.shape[1]), dtype=np.int32)
    cv2.fillPoly(mask, np.int32([pts]), 1)
    mask = mask.astype(bool)
    # Fill in polygon with image
    out = np.zeros_like(src)
    out[mask] = src[mask]

    cropped_image = out[int(face_area[0].top()):int(face_area[0].bottom()),
                    int(face_area[0].left()):int(face_area[0].right())]
    cropped_image = cv2.resize(cropped_image, (130, 130))

    return LBP_ORB_Vectors(cropped_image, orb)

def get_face_points_array(without_part, shape):
    if without_part is None or "full":
        pts = np.array(  # the 27 points on the face
            [[shape.part(0).x, shape.part(0).y], [shape.part(1).x, shape.part(1).y],
             [shape.part(2).x, shape.part(2).y], [shape.part(3).x, shape.part(3).y],
             [shape.part(4).x, shape.part(4).y], [shape.part(5).x, shape.part(5).y],
             [shape.part(6).x, shape.part(6).y], [shape.part(7).x, shape.part(7).y],
             [shape.part(8).x, shape.part(8).y], [shape.part(9).x, shape.part(9).y],
             [shape.part(10).x, shape.part(10).y], [shape.part(11).x, shape.part(11).y],
             [shape.part(12).x, shape.part(12).y], [shape.part(13).x, shape.part(13).y],
             [shape.part(14).x, shape.part(14).y], [shape.part(15).x, shape.part(15).y],
             [shape.part(16).x, shape.part(16).y], [shape.part(26).x, shape.part(26).y],
             [shape.part(25).x, shape.part(25).y], [shape.part(24).x, shape.part(24).y],
             [shape.part(23).x, shape.part(23).y], [shape.part(22).x, shape.part(22).y],
             [shape.part(27).x, shape.part(27).y], [shape.part(21).x, shape.part(21).y],
             [shape.part(20).x, shape.part(20).y], [shape.part(19).x, shape.part(19).y],
             [shape.part(18).x, shape.part(18).y], [shape.part(17).x, shape.part(17).y]],
            dtype=np.int32)
    elif "eyebrows" == without_part:
        pts = np.array(  # the 27 points on the face
            [[shape.part(0).x, shape.part(0).y], [shape.part(1).x, shape.part(1).y],
             [shape.part(2).x, shape.part(2).y], [shape.part(3).x, shape.part(3).y],
             [shape.part(4).x, shape.part(4).y], [shape.part(5).x, shape.part(5).y],
             [shape.part(6).x, shape.part(6).y], [shape.part(7).x, shape.part(7).y],
             [shape.part(8).x, shape.part(8).y], [shape.part(9).x, shape.part(9).y],
             [shape.part(10).x, shape.part(10).y], [shape.part(11).x, shape.part(11).y],
             [shape.part(12).x, shape.part(12).y], [shape.part(13).x, shape.part(13).y],
             [shape.part(14).x, shape.part(14).y], [shape.part(15).x, shape.part(15).y],
             [shape.part(16).x, shape.part(16).y], [shape.part(45).x, shape.part(45).y],
             [shape.part(44).x, shape.part(44).y], [shape.part(43).x, shape.part(43).y],
             [shape.part(27).x, shape.part(27).y], [shape.part(38).x, shape.part(38).y],
             [shape.part(37).x, shape.part(37).y], [shape.part(36).x, shape.part(36).y]],
            dtype=np.int32)
    elif "eyes" == without_part:
        pts = np.array(  # the 27 points on the face
            [[shape.part(0).x, shape.part(0).y], [shape.part(1).x, shape.part(1).y],
             [shape.part(2).x, shape.part(2).y], [shape.part(3).x, shape.part(3).y],
             [shape.part(4).x, shape.part(4).y], [shape.part(5).x, shape.part(5).y],
             [shape.part(6).x, shape.part(6).y], [shape.part(7).x, shape.part(7).y],
             [shape.part(8).x, shape.part(8).y], [shape.part(9).x, shape.part(9).y],
             [shape.part(10).x, shape.part(10).y], [shape.part(11).x, shape.part(11).y],
             [shape.part(12).x, shape.part(12).y], [shape.part(13).x, shape.part(13).y],
             [shape.part(14).x, shape.part(14).y], [shape.part(15).x, shape.part(15).y],
             [shape.part(16).x, shape.part(16).y], [shape.part(26).x, shape.part(26).y],
             [shape.part(25).x, shape.part(25).y], [shape.part(24).x, shape.part(24).y],
             [shape.part(23).x, shape.part(23).y], [shape.part(22).x, shape.part(22).y],
             [shape.part(42).x, shape.part(42).y], [shape.part(43).x, shape.part(43).y],
             [shape.part(44).x, shape.part(44).y], [shape.part(45).x, shape.part(45).y],
             [shape.part(46).x, shape.part(46).y], [shape.part(47).x, shape.part(47).y],
             [shape.part(42).x, shape.part(42).y], [shape.part(22).x, shape.part(22).y],
             [shape.part(27).x, shape.part(27).y],
             [shape.part(21).x, shape.part(21).y], [shape.part(39).x, shape.part(39).y],
             [shape.part(40).x, shape.part(40).y], [shape.part(41).x, shape.part(41).y],
             [shape.part(36).x, shape.part(36).y], [shape.part(37).x, shape.part(37).y],
             [shape.part(38).x, shape.part(38).y], [shape.part(39).x, shape.part(39).y],
             [shape.part(21).x, shape.part(21).y], [shape.part(20).x, shape.part(20).y],
             [shape.part(19).x, shape.part(19).y], [shape.part(18).x, shape.part(18).y],
             [shape.part(17).x, shape.part(17).y]],
            dtype=np.int32)
    elif "nose" == without_part:
        pts = np.array(  # the 27 points on the face
            [[shape.part(0).x, shape.part(0).y], [shape.part(1).x, shape.part(1).y],
             [shape.part(2).x, shape.part(2).y], [shape.part(3).x, shape.part(3).y],
             [shape.part(4).x, shape.part(4).y], [shape.part(5).x, shape.part(5).y],
             [shape.part(6).x, shape.part(6).y], [shape.part(7).x, shape.part(7).y],
             [shape.part(8).x, shape.part(8).y], [shape.part(9).x, shape.part(9).y],
             [shape.part(10).x, shape.part(10).y], [shape.part(11).x, shape.part(11).y],
             [shape.part(12).x, shape.part(12).y], [shape.part(13).x, shape.part(13).y],
             [shape.part(14).x, shape.part(14).y], [shape.part(15).x, shape.part(15).y],
             [shape.part(16).x, shape.part(16).y], [shape.part(26).x, shape.part(26).y],
             [shape.part(25).x, shape.part(25).y], [shape.part(24).x, shape.part(24).y],
             [shape.part(23).x, shape.part(23).y], [shape.part(22).x, shape.part(22).y],
             [shape.part(27).x, shape.part(27).y],

             [shape.part(28).x, shape.part(28).y], [shape.part(29).x, shape.part(29).y],
             [shape.part(30).x, shape.part(30).y], [shape.part(35).x, shape.part(35).y],
             [shape.part(34).x, shape.part(34).y], [shape.part(33).x, shape.part(33).y],
             [shape.part(32).x, shape.part(32).y], [shape.part(31).x, shape.part(31).y],
             [shape.part(30).x, shape.part(30).y], [shape.part(29).x, shape.part(29).y],
             [shape.part(28).x, shape.part(28).y], [shape.part(27).x, shape.part(27).y],

             [shape.part(21).x, shape.part(21).y],
             [shape.part(20).x, shape.part(20).y], [shape.part(19).x, shape.part(19).y],
             [shape.part(18).x, shape.part(18).y], [shape.part(17).x, shape.part(17).y]],
            dtype=np.int32)
    elif "mouth" == without_part:
        # print("HI")
        pts = np.array(  # the 27 points on the face
            [[shape.part(0).x, shape.part(0).y],

             [shape.part(48).x, shape.part(48).y], [shape.part(49).x, shape.part(49).y],
             [shape.part(50).x, shape.part(50).y], [shape.part(51).x, shape.part(51).y],
             [shape.part(52).x, shape.part(52).y], [shape.part(53).x, shape.part(53).y],
             [shape.part(54).x, shape.part(54).y], [shape.part(55).x, shape.part(55).y],
             [shape.part(56).x, shape.part(56).y], [shape.part(57).x, shape.part(57).y],
             [shape.part(58).x, shape.part(58).y], [shape.part(59).x, shape.part(59).y],
             [shape.part(60).x, shape.part(60).y], [shape.part(48).x, shape.part(48).y],
             [shape.part(0).x, shape.part(0).y],

             [shape.part(1).x, shape.part(1).y],
             [shape.part(2).x, shape.part(2).y], [shape.part(3).x, shape.part(3).y],
             [shape.part(4).x, shape.part(4).y], [shape.part(5).x, shape.part(5).y],
             [shape.part(6).x, shape.part(6).y], [shape.part(7).x, shape.part(7).y],
             [shape.part(8).x, shape.part(8).y], [shape.part(9).x, shape.part(9).y],
             [shape.part(10).x, shape.part(10).y], [shape.part(11).x, shape.part(11).y],
             [shape.part(12).x, shape.part(12).y], [shape.part(13).x, shape.part(13).y],
             [shape.part(14).x, shape.part(14).y], [shape.part(15).x, shape.part(15).y],
             [shape.part(16).x, shape.part(16).y], [shape.part(26).x, shape.part(26).y],
             [shape.part(25).x, shape.part(25).y], [shape.part(24).x, shape.part(24).y],
             [shape.part(23).x, shape.part(23).y], [shape.part(22).x, shape.part(22).y],
             [shape.part(27).x, shape.part(27).y], [shape.part(21).x, shape.part(21).y],
             [shape.part(20).x, shape.part(20).y], [shape.part(19).x, shape.part(19).y],
             [shape.part(18).x, shape.part(18).y], [shape.part(17).x, shape.part(17).y]],
            dtype=np.int32)

    return pts

def LBP_ORB_Vectors(cropped_image, ORB_Object):
    # Read file
    src = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Calculate local binary patterns
    lbp_image = local_binary_pattern(src, 8, 2, method='nri_uniform')
    # Find histogram of the uniform values
    histogram = np.unique(lbp_image, return_counts=True)
    # print(histogram)
    lbp_vector = [a for a in histogram[1]]

    # find the keypoints with ORB
    kp = ORB_Object.detect(src, None)
    # compute the descriptors with ORB
    kp, des = ORB_Object.compute(src, kp)

    # Return list
    # print(emotion)
    return [lbp_vector, des[0].tolist()]  # Maybe it's not des[0]?

def processed_feature_list(list): # list = [lbp, orb]
    vector_LBP = list[0]
    vector_ORB = list[1]
    max_lbp = max(vector_LBP)
    vector_LBP = [a / max_lbp for a in vector_LBP]

    max_orb = max(vector_ORB)
    vector_ORB = [a / max_orb for a in vector_ORB]
    # print(val[0])
    vector = vector_LBP + vector_ORB

    c = 1e-5
    avg = sum(vector) / len(vector)
    r = 0
    for a in vector:
        r += (a - avg) ** 2
    return [100 * (a - avg) / (r + c) for a in vector]

def init_model():
    model_abs_path = "../Polygence-Jupyter/model.dat"
    model = cv2.ml.SVM_load(model_abs_path)
    return model

model = init_model()

number_to_emotion_dict = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear", 4: "happiness", 5: "neutral", 6: "sadness",
            7: "surprise"}

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    print("Pepega")

counter = 0
while rval:
    counter +=1
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    #if counter // 100 == 0:


    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == ord(" "): # Take photo on space
        vector = crop_and_vector(frame, "full")
        processed_vector = processed_feature_list(vector)
        pred = int(model.predict(np.matrix(processed_vector, dtype=np.float32))[1])
        print("Prediction: " + str(number_to_emotion_dict[pred]))

vc.release()
cv2.destroyWindow("preview")

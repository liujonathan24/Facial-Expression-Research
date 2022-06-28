#ORB
import os
import dlib
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

list_of_lists = []
for a in ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]:
    faces_folder_path = "./New-CK+/" + a
    for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
        print("Processing file: {}".format(f))
        src = cv2.imread(f, 0)
        src = src[1:129, 1:129]
        print(src.shape)
        #print(len(src[0]))
        print('hi')
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(src, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(src, kp)

        list_of_lists.append([a, des[0]])

        # https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html

        # draw only keypoints location,not size and orientation
        #img2 = cv2.drawKeypoints(src, kp, None, color=(0, 255, 0), flags=0)
        #plt.imshow(img2), plt.show()



vector_dataframe = pd.DataFrame(data=np.array(list_of_lists), columns=['Emotion', 'ORB_Vector'])
print(vector_dataframe)
vector_dataframe.to_json("./ORB_vector_dataframe.json", orient='columns')

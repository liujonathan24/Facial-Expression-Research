# LBP
import os
import glob
import cv2
from skimage.feature import local_binary_pattern
import scipy
import pandas as pd
import numpy as np


emotions_list = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
list_of_lists = []
for a in emotions_list:
    faces_folder_path = "./New-CK+/" + str(a)

    for pic_file in glob.glob(os.path.join(faces_folder_path, "*.png")):
        src = cv2.imread(pic_file, 0)
        lbp_image = local_binary_pattern(src, 8, 2, method='nri_uniform')
        histogram = scipy.stats.itemfreq(lbp_image)
        vector = []
        for c, b in histogram:
            vector.append(b)
        list_of_lists.append([a, vector])

vector_dataframe = pd.DataFrame(data=np.array(list_of_lists), columns=['Emotion', 'Vector'])
print(vector_dataframe)
vector_dataframe.to_json("./vector_dataframe.json", orient='columns')

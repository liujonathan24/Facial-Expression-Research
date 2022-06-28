import cv2
import mediapipe as mp
import glob
import os
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
photo_landmark_dict = {}
list_paths = []
shapes = []
photo_counter = 0

# CK+ load
for emotion in ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]:
    emo_folder_path = "../Datasets/CK+/" + emotion
    for pic_file in glob.glob(os.path.join(emo_folder_path, "*png")):
        list_paths.append(pic_file)
        photo_landmark_dict[photo_counter] = [emotion]
        photo_counter += 1

# JAFFE load
for pic_file in glob.glob(os.path.join("../Datasets/JAFFE", "*tiff")):
    list_paths.append(pic_file)
    if "AN" in pic_file:
        photo_landmark_dict[photo_counter] = ["anger"]
    elif "DI" in pic_file:
        photo_landmark_dict[photo_counter] = ["disgust"]
    elif "FE" in pic_file:
        photo_landmark_dict[photo_counter] = ["fear"]
    elif "HA" in pic_file:
        photo_landmark_dict[photo_counter] = ["happiness"]
    elif "NE" in pic_file:
        photo_landmark_dict[photo_counter] = ["neutral"]
    elif "SA" in pic_file:
        photo_landmark_dict[photo_counter] = ["sadness"]
    else:
        photo_landmark_dict[photo_counter] = ["surprise"]
    photo_counter += 1

# FER-2013
FER_emo_dict = {0:"anger", 1:"disgust", 2:"fear", 3:"happiness", 4:"neutral", 5:"sadness", 6:"surprise"}
for set in ["test/", "train/"]:
    root = "../Datasets/FER2013/" + set
    for index, emotion in enumerate(["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]):
        emo_folder_path = root + emotion
        for pic_file in glob.glob(os.path.join(emo_folder_path, "*jpg")):
            list_paths.append(pic_file)
            photo_landmark_dict[photo_counter] = [FER_emo_dict[index]]
            photo_counter += 1
print(photo_counter)

skipped = 0

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2) as face_mesh:
    for idx, file in enumerate(list_paths):
        image = cv2.imread(file, 0)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks is not None:
            x_coords = []
            y_coords = []
            z_coords = []
            for count, mark in enumerate(results.multi_face_landmarks[0].landmark):
                x_coords.append(mark.x)
                y_coords.append(mark.y)
                z_coords.append(mark.z)

                # MUST CROP PHOTO FIRST AND THEN STUFF.
            shape = image.shape

            x_coords = [x * shape[0] for x in x_coords]
            y_coords = [y * shape[1] for y in y_coords]

            range_x = max(x_coords) - min(x_coords)
            range_y = max(y_coords) - min(y_coords)
            offset_x = [x - min(x_coords) for x in x_coords]
            offset_y = [y - min(y_coords) for y in y_coords]
            new_x = [x / range_x for x in offset_x]
            new_y = [y / range_y for y in offset_y]

            photo_landmark_dict[idx][1:479] = zip(new_x, new_y, z_coords)
        else:
            skipped += 1
        if idx % 200 == 0:
            print(idx)
            print(photo_landmark_dict[idx])

print(skipped)
#print(len(photo_landmark_dict[0]))



dataframe = pd.DataFrame(data=photo_landmark_dict)
print(dataframe.head())
print(dataframe.tail())
dataframe.to_json("./photo_landmark_list.json", orient='columns')
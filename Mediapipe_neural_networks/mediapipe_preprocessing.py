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
for set in ["test", "train"]:
    root = "../Datasets/FER2013/" + set
    for emotion in ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]:
        emo_folder_path = root + emotion
        for pic_file in glob.glob(os.path.join(emo_folder_path, "*jpg")):
            list_paths.append(pic_file)
            photo_landmark_dict[photo_counter] = [emotion]
            photo_counter += 1

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(list_paths):
    image = cv2.imread(file)
    # MUST CROP PHOTO FIRST AND THEN STUFF.
    shape = image.shape
    shapes.append(shape)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    list = np.array([])
    for count, mark in enumerate(results.multi_face_landmarks[0].landmark):
        coords = [mark.x, mark.y, mark.z]
        photo_landmark_dict[idx].append(coords)

#print(len(photo_landmark_dict[0]))
# Edit percents to remove "offset" -> cropping image
def crop_pictures(photo_landmark_dict, shapes):
    for picture_count in range(len(photo_landmark_dict.keys())): # Replace with range(len(photo_landmark_dict.keys()))
        x_coords, y_coords, z_coords  = zip(*photo_landmark_dict[picture_count][1:479])

        x_coords = [x * shapes[picture_count][0] for x in x_coords]
        y_coords = [y * shapes[picture_count][1] for y in y_coords]

        range_x = max(x_coords) - min(x_coords)
        range_y = max(y_coords) - min(y_coords)
        offset_x = [x - min(x_coords) for x in x_coords]
        offset_y = [y - min(y_coords) for y in y_coords]
        new_x = [x/range_x for x in offset_x]
        new_y = [y/range_y for y in offset_y]

        photo_landmark_dict[picture_count][1:479] = zip(new_x, new_y, z_coords)
    return photo_landmark_dict

photo_landmark_dict = crop_pictures(photo_landmark_dict, shapes)

dataframe = pd.DataFrame(data=photo_landmark_dict)
print(dataframe.head())
print(dataframe.tail())
dataframe.to_json("./photo_landmark_list.json", orient='columns')
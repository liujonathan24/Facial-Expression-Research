import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import numpy as np
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


photo_landmark_dict = {"Emotion":[]}
for a in range(478):
  photo_landmark_dict[a] = []


# For static images:
#IMAGE_FILES = ["../CK+/anger/S010_004_00000019.png"]
list_paths = []
folder = "../CK+"
for emotion in ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]:
  emo_folder_path = folder + "/" + emotion
  #print("elp_0")
  for pic_file in glob.glob(os.path.join(emo_folder_path, "*png")):
      list_paths.append(pic_file)
      photo_landmark_dict["Emotion"].append(emotion)



for pic_file in glob.glob(os.path.join("../JAFFE", "*tiff")):
    list_paths.append(pic_file)
    if "AN" in pic_file:
        photo_landmark_dict["Emotion"].append("anger")
    elif "DI" in pic_file:
        photo_landmark_dict["Emotion"].append("disgust")
    elif "FE" in pic_file:
        photo_landmark_dict["Emotion"].append("fear")
    elif "HA" in pic_file:
        photo_landmark_dict["Emotion"].append("happiness")
    elif "NE" in pic_file:
        photo_landmark_dict["Emotion"].append("neutral")
    elif "SA" in pic_file:
        photo_landmark_dict["Emotion"].append("sadness")
    else:
        photo_landmark_dict["Emotion"].append("surprise")

print(list_paths)




drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
#print("1")
  for idx, file in enumerate(list_paths):
    print(file)
    image = cv2.imread(file)
    # MUST CROP PHOTO FIRST AND THEN STUFF.
    shape = image.shape
    print(shape)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    list = np.array([])
    for count, idx in enumerate(results.multi_face_landmarks[0].landmark):
      photo_landmark_dict[count].append([idx.x, idx.y, idx.z])
    break
photo_landmark_dict["Emotion"] = "anger"
print(len(photo_landmark_dict["Emotion"]))
print(len(photo_landmark_dict[1]))

# Edit percents to remove "offset" -> cropping image
for landmark in range(378):
    photo_landmark_dict[landmark]
x_coords = [photo_landmark_dict[a][0] * shape[0] for a in range(378)]
x_range = max(x_coords) - min(x_coords)
x_offset_coords = [x - min(x_coords) for x in x_coords]
final_x = [x/x_range for x in x_offset_coords]

y_coords = [photo_landmark_dict[a][1] * shape[1] for a in range(378)]
y_range = max(y_coords) - min(y_coords)
y_offset_coords = [y - min(y_coords) for y in y_coords]
final_y = [y/y_range for y in y_offset_coords]

z_percent = [photo_landmark_dict[a][2] for a in range(378)]

cropped_landmark_list = list(zip(final_x, final_y, z_percent))

for a in range(378): # Placing cropped coordniates back into the dict.
    photo_landmark_dict[a] = cropped_landmark_list[a]

dataframe = pd.DataFrame(data=photo_landmark_dict)
dataframe.head()
dataframe.to_json("./photo_landmark_list.json", orient='columns')
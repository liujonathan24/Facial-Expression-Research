import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = ["../CK+/anger/S010_004_00000019.png"]

folder_path = "../CK+"
for emotion in ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]:
        emo_folder_path = folder_path + "/" + emotion
        #print("elp_0")
        for pic_file in glob.glob(os.path.join(emo_folder_path, "*png")):




photo_landmark_list = np.array([])


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
#print("1")
  for idx, file in enumerate(IMAGE_FILES):
    print(file)
    image = cv2.imread(file)
    # MUST CROP PHOTO FIRST AND THEN STUFF.
    shape = image.shape
    print(shape)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    list = np.array([])
    for count, idx in enumerate(results.multi_face_landmarks[0].landmark):
      np.append(list, [idx.x, idx.y, idx.z])
    np.append(photo_landmark_list, list)



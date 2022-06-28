import mediapipe as mp
import cv2
import math
import glob
import os
import numpy as np

faces_folder_path = "../CK+/"
emos = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def resize_and_show(name, image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow(name, img)

aaaa = cv2.imread("../CK+/anger/S010_004_00000019.png")
cv2.imshow("elp", aaaa)
cv2.waitKey(0)
quit()

# Read images with OpenCV.
abcd = [cv2.imread(name) for a in emos for name in glob.glob(os.path.join(faces_folder_path+a, "*.png"))]
print(abcd[0])
cv2.imshow("elp", abcd[0])
quit()
images = {name: cv2.imread(name) for a in emos for name in glob.glob(os.path.join(faces_folder_path+a, "*.png"))}

# Preview the images.
for name, image in images.items():
    print(name)
    resize_and_show(name, image)

#mp_face_mesh = mp.solutions.face_mesh

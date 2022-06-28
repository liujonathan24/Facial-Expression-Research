import cv2
import numpy as np
import mediapipe as mp
from Points import Points
import torch
import torch.nn as nn
import torch.nn.functional as F
#from nn_max_points import Model

# PREPARATION VARIABLES, CLASSES, AND FUNCTIONS
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(268, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = x.view(-1, 268)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x
def init_model():
    model = Model(268)
    dict = torch.load("/Users/Fritz/Documents/Polygence/Mediapipe_experiments/model_v1.pt")
    model.load_state_dict(dict)

    model.eval()

    return model
def landmarks(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = []
        for count, mark in enumerate(results.multi_face_landmarks[0].landmark):
            coords = [mark.x, mark.y, mark.z]
            landmarks.append(coords)
    return landmarks
def crop_picture(landmarks, shape):
    x_coords, y_coords, z_coords  = zip(*landmarks[1:479])

    x_coords = [x * shape[0] for x in x_coords]
    y_coords = [y * shape[1] for y in y_coords]
    print(x_coords)
    print(type(x_coords))
    range_x = max(x_coords) - min(x_coords)
    range_y = max(y_coords) - min(y_coords)
    offset_x = [x - min(x_coords) for x in x_coords]
    offset_y = [y - min(y_coords) for y in y_coords]
    new_x = [x/range_x for x in offset_x]
    new_y = [y/range_y for y in offset_y]

    landmarks[1:479] = zip(new_x, new_y, z_coords)
    return landmarks
def set_of_landmarks(landmarks, set="max"):
    new_landmarks = []
    for point in Points[set].value:
        new_landmarks.extend(landmarks[point][0:2])
    return new_landmarks

model = init_model()
number_to_emotion_dict = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear", 4: "happiness", 5: "neutral", 6: "sadness",
            7: "surprise"}


# WINDOW EXECUTION

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

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
        vector = landmarks(frame)
        cropped_vector = crop_picture(vector, frame.shape)
        selected_cropped_vector = set_of_landmarks(cropped_vector, "max")
        array = np.array(selected_cropped_vector, dtype=np.float32).reshape((268, 1))
        tensor = torch.from_numpy(array)
        pred = model(tensor)
        back_to_array = pred.detach().numpy()
        print(back_to_array)
        max = 0
        max_index = 100
        for index in range(8):
            if back_to_array[0][index] > max:
                max = back_to_array[0][index]
                max_index = index

        print("Prediction: " + str(number_to_emotion_dict[max_index]))

vc.release()
cv2.destroyWindow("preview")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Points import Points
from sklearn.model_selection import train_test_split

# model_v1 = 63.72%

emotion_dict = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3, "happiness": 4, "neutral": 5, "sadness": 6,"sad": 6,
            "surprise": 7}

def read_data(path, label_points):
    print(len(label_points))
    landmarks = pd.read_json(path)
    emotions = np.asarray([emotion_dict[y] for y in landmarks.iloc[0].astype(object)], dtype=np.float32)
    print(emotions)
    columns = landmarks.shape[1]
    #points = np.zeros((1133, len(label_points) * 2), dtype=np.float32)
    points = []
    for photo in range(columns):
        #temp_list = [landmarks[photo].iloc[point+1][0:2] for point in label_points]
        #print(temp_list)
        for point in label_points:
            coords = landmarks[photo].iloc[point+1][0:2]
            points.extend(coords)

    points = np.array(points, dtype=np.float32).reshape(1133, len(label_points*2))

    print(f"shape at the end: {points.shape}")
    return emotions, points
    #return [[a, b] for (a, b) in zip(np.concatenate((JAFFE_EMO, CK_EMO)), np.concatenate((JAFFE_VEC, CK_VEC)))] #
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(268, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = x.view(-1, 268)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x

min_wanted_points = Points.right_eye_middle.value + Points.left_eye_middle.value + Points.nose.value + Points.mouth_inner.value
max_wanted_points = min_wanted_points + Points.right_eye_inner.value + Points.right_eye_outer.value + \
    Points.left_eye_inner.value + Points.left_eye_outer.value

full_y, full_x = read_data("./photo_landmark_list.json", max_wanted_points)

X_test = torch.from_numpy(full_x)
y_test = torch.from_numpy(full_y)
y_test = y_test.view(y_test.shape[0], 1)

complete_y_test = np.zeros((1133, 7))
for num, label in enumerate(y_test):
    if label != 5:
        complete_y_test[num][int(label)] = 10


model = Model(268)
dict = torch.load("/Users/Fritz/Documents/Polygence/Mediapipe_experiments/model-v3.pt")
model.load_state_dict(dict)
model.eval()

with torch.no_grad():
    emotions = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]} # emotion:[num correct, total]
    print(y_test)
    #for val in range(y_test.shape[0]):
    #    emotions[int(y_test[val])][1] += 1

    y_predicted = model(X_test)

    correct = 0
    for row in range(y_predicted.shape[0]):
        dist = 100
        closest_index = None
        for index in range(7):
            if abs(float(y_predicted[row][index]) - 1) < dist:
                dist = abs(float(y_predicted[row][index]) - 1)
                closest_index = index
        emotions[int(y_test[row])][1] += 1
        if closest_index == y_test[row]:
            correct += 1
            emotions[int(y_test[row])][0] += 1

    print(f"correct: {correct}")
    print(f"total: {y_predicted.shape[0]}")
    print(f"Emotions: {emotions}")


    '''
    bool_array = y_predicted_cls.eq(torch.from_numpy(complete_y_test))
    counter = 0
    for row in range(bool_array.shape[0]):
        if True in bool_array[row]:
            counter +=1
        for index in range(8):
            if bool_array[row][index] is True:
                if index == 0:
                    print(f"row: {row}, index: {index}")
                emotions[index][0] += 1
                break
    print(f"counter: {counter}")
    print(emotions)


    print(y_predicted_cls.eq(torch.from_numpy(complete_y_test)).sum())
    print(float(complete_y_test.shape[0]))
    acc = y_predicted_cls.eq(torch.from_numpy(complete_y_test)).sum() / float(complete_y_test.shape[0])
    #print(f"number correct which was neutral: {neutral_correct_counter}")
    #print(f"number of neutral: {number_neutral}")
    #print(y_predicted_cls.eq(y_test).sum() -neutral_correct_counter)
    #print((float(y_test.shape[0])-number_neutral))
    #print(f"percent without neutral: {(y_predicted_cls.eq(y_test).sum() -neutral_correct_counter) / (float(y_test.shape[0])-number_neutral)}")
    


    print(f'accuracy: {acc.item():.4f}')'''


# Version 1: 38.60 accuracy (exactly 1 epoch)
#v1: 80.70%
#v2:

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Points import Points
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import chain

emotion_dict = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3, "happiness": 4, "neutral": 5, "sadness": 6,"sad": 6,
            "surprise": 6}

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


min_wanted_points = Points.right_eye_middle.value + Points.left_eye_middle.value + Points.nose.value + Points.mouth_inner.value
max_wanted_points = min_wanted_points + Points.right_eye_inner.value + Points.right_eye_outer.value + \
    Points.left_eye_inner.value + Points.left_eye_outer.value

full_y, full_x = read_data("./photo_landmark_list.json", max_wanted_points)

X_train, X_test, y_train, y_test = train_test_split(full_x, full_y, test_size=0.05, random_state=1234)


X_train = torch.from_numpy(X_train)
print(f"X_train type: {X_train.shape}")
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#print(X_train[0])
#print(X_test[0])
print(y_train.shape)
print(y_test.shape)
print(f"y_test[0] = {y_test[0]}")

# 1) Model
# Linear model f = wx + b , sigmoid at the end
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

model = Model(268)

# 2) Loss and optimizer
num_epochs = 1000
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3) Training loop
counter = 0
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)
    #print(f"y_pred shape: {y_pred.view(-1).shape}")
    #print(f"y_pred type: {type(y_pred.view(-1))}")
    #print(f"y_pred = {y_pred}")
    new_label = np.zeros((1076, 7))
    for num, label in enumerate(y_train):
        if label != 5:
            new_label[num][int(label)] = 10
    #print(f"label shape: {torch.from_numpy(new_label).shape}")
    #print(f"label type: {type(torch.from_numpy(new_label))}")
    new_pred = y_pred.view(1076, 7)
    new_label = torch.from_numpy(new_label).float().view(1076,7)
    loss = criterion(new_pred, new_label)


    # Backward pass and update
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')



with torch.no_grad():
    correct = 0
    y_predicted = model(X_test)
    for row in range(y_predicted.shape[0]):
        dist = 100
        closest_index = None
        for index in range(7):
            if abs(float(y_predicted[row][index])-1) < dist:
                dist = abs(float(y_predicted[row][index])-1)
                closest_index = index
        if closest_index == y_test[row]:
            correct += 1

    print(f"correct: {correct}")
    print(f"total: {y_predicted.shape[0]}")

torch.save(model.state_dict(), "./model-v10.pt")
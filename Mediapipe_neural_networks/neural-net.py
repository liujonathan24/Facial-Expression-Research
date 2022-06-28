# Version 1: 31.72% accuracy (exactly 1 epoch)
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


min_wanted_points = Points.right_eye_middle.value + Points.left_eye_middle.value + Points.nose.value + Points.mouth_inner.value
max_wanted_points = min_wanted_points + Points.right_eye_inner.value + Points.right_eye_outer.value + \
    Points.left_eye_inner.value + Points.left_eye_outer.value

full_y, full_x = read_data("./photo_landmark_list.json", min_wanted_points)

X_train, X_test, y_train, y_test = train_test_split(full_x, full_y, test_size=0.2, random_state=1234)


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
        self.fc1 = nn.Linear(140, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = x.view(-1, 140)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x

model = Model(140)

# 2) Loss and optimizer
num_epochs = 20
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(zip(X_train, y_train)):
        # Forward pass and loss
        y_pred = model(image)
        #print(f"y_pred shape: {y_pred.view(-1).shape}")
        #print(f"y_pred type: {type(y_pred.view(-1))}")
        #print(f"y_pred = {y_pred}")
        new_label = np.zeros(8)
        new_label[int(label)] = 1
        #print(f"label shape: {torch.from_numpy(new_label).shape}")
        #print(f"label type: {type(torch.from_numpy(new_label))}")
        loss = criterion(y_pred.view(8, 1), torch.from_numpy(new_label).float().view(8,1))

        optimizer.zero_grad()

        # Backward pass and update
        loss.backward()
        optimizer.step()


    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
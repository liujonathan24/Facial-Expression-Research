import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

emotion_dict = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3, "happiness": 4, "neutral": 5, "sadness": 6,"sad": 6,
            "surprise": 7}

def read_data(parts):
    JAFFE = pd.read_json(parts + '-JAFFE-vectors.json')
    CK = pd.read_json(parts + '-CK+-vectors.json')

    JAFFE_EMO = np.asarray([emotion_dict[y] for y in JAFFE['Emotion'].astype(object)], dtype=np.float32)
    JAFFE_VEC = np.array([np.array(x, dtype=np.float32) for x in JAFFE['Final_Vector']])
    CK_EMO = np.asarray([emotion_dict[y] for y in CK['Emotion'].astype(object)], dtype=np.float32)
    CK_VEC = np.array([np.array(x, dtype=np.float32) for x in CK['Final_Vector']])
    print(len(JAFFE_EMO))
    print(len(CK_EMO))
    return np.concatenate((JAFFE_EMO, CK_EMO)), np.concatenate((JAFFE_VEC, CK_VEC))
    #return [[a, b] for (a, b) in zip(np.concatenate((JAFFE_EMO, CK_EMO)), np.concatenate((JAFFE_VEC, CK_VEC)))]



full_y, full_x = read_data("../Polygence-Jupyter/full")

#print(full_y)
#full_y = [emotion_dict[y] for y in full_y]
#print(full_y)
#print(full_x.shape)
n_samples, n_features = full_x.shape, 91

X_train, X_test, y_train, y_test = train_test_split(full_x, full_y, test_size=0.2, random_state=1234)

# scale
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 1) Model
# Linear model f = wx + b , sigmoid at the end
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(91, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.model(x)

model = Model(n_features)

# 2) Loss and optimizer
num_epochs = 100000
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 1000 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
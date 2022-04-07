import cv2 as cv
import numpy as np
import pandas as pd


vectors = pd.read_json("./Final_Vectors.json")
label_emotions = []
data_vector_list = []
emo_dict = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3, "happiness": 4, "neutral": 5, "sadness": 6, "surprise": 7}

shift = 5

for a in range(920):
    # exclude 5% of each?
    # emo_dict_count = {'anger': 45, 'contempt': 18, 'disgust': 59, 'fear': 25, 'happiness': 69, 'neutral': 593,
    # 'sadness': 28, 'surprise': 83}
    if a in list(range(-shift, 920, 20)):
        continue
    emotion, vector = list(vectors.iloc[a])
    data_vector_list.append(vector)
    label_emotions.append(emo_dict[emotion])

labels = np.array(label_emotions, dtype=int)
trainingData = np.matrix(data_vector_list, dtype=np.float32)
print(trainingData.dtype)

# Train the SVM
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_NU_SVC)
#svm.setKernel(cv.ml.SVM_LINEAR)
#svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
#svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C -> 40%

#svm.setKernel(cv.ml.SVM_CHI2)
#svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
#svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C-> 43-44.68%

#svm.setKernel(cv.ml.SVM_RBF)
#svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
#svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C->.869, .872, .893

#svm.setKernel(cv.ml.SVM_INTER)
#svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
#svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C-> .723, .630, .808, .82978,

#svm.setKernel(cv.ml.SVM_CUSTOM)
#svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
#svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C-> ,869, .8723,.8936, .8936

svm.setKernel(cv.ml.SVM_LINEAR)
svm.setNu(.5)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
print('Before training')
svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # broken or something, returns: error: (-211:One of the arguments'
# values is out of range) The parameter nu must be between 0 and 1 in function 'checkParams'
# 'checkParams' function clearly doesn't exist though
print('After training')

ratio = {"Correct": 0, "Wrong": 0}
for a in range(-shift, 920, 20):
    print("In for loop")
    real_a = a - (a+shift+1)//20
    predit = np.matrix(list(vectors.iloc[real_a])[1], dtype=np.float32) # Prediction array
    print('between predit, predict')
    response = svm.predict(predit)[1] # Actual prediction ---- causes SIGSEGV
    # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
    print('After prediction')
    correct = label_emotions[real_a] # Correct value
    if int(response) == int(correct):
        ratio["Correct"] += 1
    else:
        ratio["Wrong"] += 1

print(ratio["Correct"])
print(ratio["Wrong"])
print(ratio["Correct"]/(ratio["Correct"] + ratio["Wrong"]))
import cv2 as cv
import numpy as np
import pandas as pd


#vectors = pd.read_json("./Polygence-Jupyter/eyebrows-CK+-vectors.json")
vectors = pd.read_json("./Polygence-Jupyter/full-CK+-vectors.json")

label_emotions = []
data_vector_list = []
emo_dict = {"anger": 0, "contempt": 1, "disgust": 2, "fear": 3, "happiness": 4, "neutral": 5, "sadness": 6,"sad": 6,
            "surprise": 7}
#shift = 5
accuracy = 0
confusion_matrix_count_total = np.zeros((8, 8))
for shift in range(5, 15):
    for a in range(920):
        # exclude 5% of each?
        # emo_dict_count = {'anger': 45, 'contempt': 18, 'disgust': 59, 'fear': 25, 'happiness': 69, 'neutral': 593,
        # 'sadness': 28, 'surprise': 83}
        # Maybe used F-1 score
        # Something which accounts for the imbalanced counts on each thing
        if a in list(range(-shift, 920, 20)):
            continue
        emotion, vector = list(vectors.iloc[a])
        data_vector_list.append(vector)
        label_emotions.append(emo_dict[emotion])

    labels = np.array(label_emotions, dtype=int)
    trainingData = np.matrix(data_vector_list, dtype=np.float32)

    # Train the SVM
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    #svm.setKernel(cv.ml.SVM_LINEAR)
    #svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    #svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C -> 40%

    #svm.setKernel(cv.ml.SVM_CHI2)
    #svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    #svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C-> 43-44.68%

    svm.setKernel(cv.ml.SVM_RBF)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C-> 86.9, 87.2, 89.3%

    #svm.setKernel(cv.ml.SVM_INTER)
    #svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    #svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C-> .723, .630, .808, .82978,

    #svm.setKernel(cv.ml.SVM_CUSTOM)
    #svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    #svm.train(trainingData, cv.ml.ROW_SAMPLE, labels) # C-> ,869, .8723,.8936, .8936
    #svm.save("/C_SVC-SVM_Linear.svm")

    #svm.setKernel(cv.ml.SVM_LINEAR)
    #svm.setNu(.5)
    #print(svm.getNu())
    #svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    #print('Before training')
    svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)
    # https://docs.opencv.org/3.4/d1/d2d/classcv_1_1ml_1_1SVM.html
    #print('After training')
    confusion_matrix_count = np.zeros((8,8))
    ratio = {"Correct": 0, "Wrong": 0}
    for a in range(-shift, 920, 20):
        #print("In for loop")
        real_a = a - (a+shift+1)//20
        predit = np.matrix(list(vectors.iloc[real_a])[1], dtype=np.float32) # Prediction array
        #print('between predit, predict')
        response = svm.predict(predit)[1]
        #print('After prediction')
        correct = label_emotions[real_a] # Correct value
        confusion_matrix_count[int(response)][correct] += 1
        confusion_matrix_count_total[int(response)][correct] += 1

        if int(response) == int(correct):
            ratio["Correct"] += 1
        else:
            ratio["Wrong"] += 1
            #print("Response was: " + str(response) + "; Response should have been " + str(correct))

    #print(ratio["Correct"])
    #print(ratio["Wrong"])
    acc = ratio["Correct"]/(ratio["Correct"] + ratio["Wrong"])
    print(acc)

    if acc > accuracy:
        accuracy = acc
        # Store it by using OpenCV functions:
        svm.save("./Polygence-Jupyter/model_PYCHARM.dat")

    confusion_matrix_count[7][0] += 1000
    print(confusion_matrix_count)

print(100 * (confusion_matrix_count_total[0][0] + confusion_matrix_count_total[1][1] +confusion_matrix_count_total[2][2] +
             confusion_matrix_count_total[3][3] +confusion_matrix_count_total[4][4] +confusion_matrix_count_total[5][5] +
             confusion_matrix_count_total[6][6] +confusion_matrix_count_total[7][7])/920)
print(confusion_matrix_count_total)
    # Explainable ML

    # Remove parts of faces, see which changes the prediction/confusion matrix the most
    # (response, correct) = (row (counting top-> bot), column(left->right))
    #
    # [ 3.  0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  1.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  3.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  1.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  3.  1.  0.  0.]
    #  [ 0.  0.  0.  0.  0. 29.  1.  1.]
    #  [ 0.  0.  0.  0.  0.  0.  0.  2.]
    #  [ 0.  0.  0.  0.  0.  0.  0.  2.]]

    # (44+15+55+18+61+581+57)/938 = 88.6%
    # [[ 44.   3.   0.   0.   0.   0.   0.   0.]
    #  [  0.  15.   4.   0.   0.   0.   0.   0.]
    #  [  0.   0.  55.   7.   0.   0.   0.   0.]
    #  [  0.   0.   0.  18.   8.   0.   0.   0.]
    #  [  0.   0.   0.   0.  61.  12.   0.   0.]
    #  [  1.   0.   0.   0.   0. 581.  28.  15.]
    #  [  0.   0.   0.   0.   0.   0.   0.  30.]
    #  [  0.   0.   0.   0.   0.   0.   0.  57.]]

    # Confusion matrix: 8x8 with the emotions -> mixups = input neutral/sad -> ends up neutral, sad, surprised

    # Mediapipe library - finding keypoints
    # use SVM on those, see how adjusting keypoints would change the classification


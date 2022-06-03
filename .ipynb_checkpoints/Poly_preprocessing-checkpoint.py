# DLib
import os
import dlib
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

for a in ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]:
    faces_folder_path = "./CK+/" + a
    for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        dets = detector(img, 1)
        # print("Number of faces detected: {}".format(len(dets))) # Should be 1
        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #   k, d.left(), d.top(), d.right(), d.bottom()))

            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
            # print(shape.part(0))
            pts = np.array(
                [[shape.part(0).x, shape.part(0).y], [shape.part(3).x, shape.part(3).y],
                 [shape.part(4).x, shape.part(4).y],
                 [shape.part(5).x, shape.part(5).y], [shape.part(6).x, shape.part(6).y],
                 [shape.part(7).x, shape.part(7).y],
                 [shape.part(8).x, shape.part(8).y], [shape.part(9).x, shape.part(9).y],
                 [shape.part(10).x, shape.part(10).y],
                 [shape.part(11).x, shape.part(11).y], [shape.part(12).x, shape.part(12).y],
                 [shape.part(13).x, shape.part(13).y],
                 [shape.part(14).x, shape.part(14).y], [shape.part(15).x, shape.part(15).y],
                 [shape.part(16).x, shape.part(16).y],
                 [shape.part(26).x, shape.part(26).y], [shape.part(25).x, shape.part(25).y],
                 [shape.part(24).x, shape.part(24).y],
                 [shape.part(23).x, shape.part(23).y], [shape.part(22).x, shape.part(22).y],
                 [shape.part(27).x, shape.part(27).y],
                 [shape.part(21).x, shape.part(21).y], [shape.part(20).x, shape.part(20).y],
                 [shape.part(19).x, shape.part(19).y],
                 [shape.part(18).x, shape.part(18).y], [shape.part(17).x, shape.part(17).y]], dtype=np.int32)

            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)

            cv2.fillPoly(mask, np.int32([pts]), 1)
            mask = mask.astype(np.bool)

            out = np.zeros_like(img)
            out[mask] = img[mask]

            src = cv2.imread(f, 0)
            cropped_image = out[int(d.top()):int(d.bottom()), int(d.left()):int(d.right())]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = cv2.resize(cropped_image, (130, 130))
            #plt.imshow(cropped_image)
            #plt.show()
            x = 7 + len(a)
            cv2.imwrite(str('New-CK+/' + a + '/'+  f[x:]), cropped_image)

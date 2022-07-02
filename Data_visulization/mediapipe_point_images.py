# import libraries
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

items = ["../Datasets/FER2013/train/angry/Training_33331.jpg", "../Datasets/CK+/anger/S010_004_00000019.png", "../Datasets/JAFFE/KA.AN1.39.tiff",
         "./annotated_image0.png", "./annotated_image1.png", "./annotated_image2.png"]

# create figure
fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 2
columns = 3

# reading images
Image1 = cv2.imread("../Datasets/FER2013/train/angry/Training_33331.jpg")
Image2 = cv2.imread("../Datasets/CK+/anger/S010_004_00000019.png")
Image3 = cv2.imread("../Datasets/JAFFE/KA.AN1.39.tiff")
Image4 = cv2.imread("./annotated_image0.png")
Image5 = cv2.imread("./annotated_image1.png")
Image6 = cv2.imread("./annotated_image2.png")

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("Sample FER-2013 Photo")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(Image2)
plt.axis('off')
plt.title("Sample CK+ Photo")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(Image3)
plt.axis('off')
plt.title("Sample JAFFE Photo")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(Image4)
plt.axis('off')
plt.title("With Mediapipe Points Drawn")

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 5)

# showing image
plt.imshow(Image5)
plt.axis('off')
plt.title("With Mediapipe Points Drawn")

# Adds a subplot at the 6th position
fig.add_subplot(rows, columns, 6)

# showing image
plt.imshow(Image6)
plt.axis('off')
plt.title("With Mediapipe Points Drawn")

plt.show()
import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline
plt.style.use('ggplot')

countries = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
FER2013 = np.array([4953, 0, 547, 5121, 8989, 6198, 6077, 4002])
CK_PLUS = np.array([45, 18, 59, 25,  69,  593, 28, 83])
JAFFE = np.array([30, 0, 29, 32, 31, 30, 31, 30])
ind = [x for x, _ in enumerate(countries)]

plt.bar(ind, JAFFE, width=0.8, label='JAFFE', color='gold', bottom=FER2013+CK_PLUS)
plt.bar(ind, CK_PLUS, width=0.8, label='CK+', color='silver', bottom=FER2013)
plt.bar(ind, FER2013, width=0.8, label='FER-2013', color='#CD853F')

plt.xticks(ind, countries)
plt.ylabel("Photo Count")
plt.xlabel("Emotions")
plt.legend(loc="upper right")

plt.title("Photo Distribution")

plt.show()
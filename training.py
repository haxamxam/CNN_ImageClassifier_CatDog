import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = r'C:\Users\your_username\Desktop\your_data_dir'
CATEGORIES = ["Cat","Dog"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  # create a path to cats and dogs directory
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img),
                               cv2.IMREAD_GRAYSCALE)  # Made it grayscale since color isn't really needed
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break

# Load up the images
#Lets look at how the data looks like
print(img_array)
print(img_array.shape) #it is a 375x500 sized picture

#Reduce the image size to consume less memory
IMG_SIZE = 50
new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # create a path to cats and dogs directory
        class_num = CATEGORIES.index(category) # We index the pics 0 and 1 where 0 is a cat and 1 is a dog
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img),
                                       cv2.IMREAD_GRAYSCALE)  # Made it grayscale since color isn't really needed
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

print(len(training_data))
print(training_data[0])
random.shuffle(training_data)

for i in training_data[:10]:
    print(i[1])

x = []
y = []

for features, labels in training_data:
    x.append(features)
    y.append(labels)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#Dumping our model into a pickle file

pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pathlib
import random
import pickle

DATADIR = 'C:\\Users\\Maith\\Documents\\Datasets\\Dataset_food-101\\images3'
data_dir = pathlib.Path(DATADIR)
Catagories = [item.name for item in data_dir.glob('*')]
print(Catagories)

IMG_SIZE = 32
'''
for category in Catagories:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        plt.imshow(img_array)
        plt.show()
        plt.imshow(new_array)
        plt.show()
        break
    break
'''
'''
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
'''

training_data = []

'''
def create_training_data():
    for category in Catagories:
        path = os.path.join(DATADIR, category)
        class_num = Catagories.index(category)
        for img in os.listdir(path):
            try:
                # add condition in this line  for creating testing dataset
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

'''

training_data = []
testing_data = []


def create_dataset():
    i = 0
    for category in Catagories:
        path = os.path.join(DATADIR, category)
        class_num = Catagories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                if i % 4 is 0:  # save 25% of data as testing data.
                    testing_data.append([new_array, class_num])
                else:
                    training_data.append([new_array, class_num])
            except Exception as e:
                pass
            i += 1


create_dataset()
print("TRAINING DATA AFTER SEPARATION", len(training_data))
print("TESTING DATA", len(testing_data))
random.shuffle(training_data)
random.shuffle(testing_data)

# for sample in training_data[:10]:
#     print(sample[1])

X_train = []
y_train = []

for features, labels in training_data:
    X_train.append(features)
    y_train.append(labels)

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = np.array(y_train).astype(int)

print("printing one sample of labels", y_train[2])

pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

X_test = []
y_test = []

for features, labels in testing_data:
    X_test.append(features)
    y_test.append(labels)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = np.array(y_test).astype(int)


pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()
pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

print("FINISHED IMPORTING DATA! ")

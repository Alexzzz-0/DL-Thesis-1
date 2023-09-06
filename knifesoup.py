import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras import layers,models


def import_dataset():

    img_list = []
    labels = []

    #获取汤的图片
    for soupname in os.listdir(r'C:\AlexZheng\soup'):
        soup_dir = 'C:\AlexZheng\soup/' + soupname
        img_soup = cv.imread(soup_dir,1)
        soup_resize = cv.resize(img_soup,(28,28),cv.INTER_AREA)
        img_list.append(soup_resize)
        labels.append(0)

    for knifename in os.listdir(r'C:\AlexZheng\knife'):
        knife_dir = 'C:\AlexZheng\knife/' + knifename
        img_knife = cv.imread(knife_dir,1)
        knife_resize = cv.resize(img_knife,(28,28),cv.INTER_AREA)
        img_list.append(knife_resize)
        labels.append(1)

    for oldman in os.listdir(r'C:\AlexZheng\oldman'):

        oldman_whole = 'C:\AlexZheng\oldman/' + oldman
        img_oldman = cv.imread(oldman_whole,1)
        oldman_resize = cv.resize(img_oldman,(28,28),cv.INTER_AREA)
        img_list.append(oldman_resize)
        labels.append(2)

    img_array = np.asarray(img_list,dtype=int)
    label_array = np.asarray(labels)

    return img_array,label_array

def shuffle_split_data(imgs,labels):
    X_train, X_test, y_trian, y_test = train_test_split(imgs, labels, test_size=0.1, random_state=58)

    return X_train, X_test, y_trian, y_test

    print(00)

def train_model(X_train,y_train):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=2)

    return model

# def test_the_img(X_test):
#
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(X_test[400])
#
#     ax.axis('off')
#     plt.show()
def main():
    img_array, label_array = import_dataset()
    X_train,X_test,y_train,y_test = shuffle_split_data(img_array,label_array)
    #test_the_img(X_test)
    model = train_model(X_train,y_train)


if __name__ == "__main__":
    main()

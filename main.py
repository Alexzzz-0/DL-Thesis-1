import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

#from tensorflow.python.keras import layers,models
from tensorflow import keras
from keras import layers,models
import tensorflow as tf

# import tensorflow.python as tf
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Flatten
# from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

#get the dataset
def load_data():
    imgs_list = []
    labels_string = []

    #prepare the labels from loading the first number of the pictures' name
    for soupname in os.listdir(r"C:\AlexZheng\soup"):

        #prepare the arrays of the pictures
        soupname_whole = 'C:\AlexZheng\soup/'+soupname
        img_soup = cv.imread(soupname_whole,1)
        imgs_list.append(img_soup)
        labels_string.append(0)

    for knifename in os.listdir(r'C:\AlexZheng\knife'):

        #get the pictures of knifes
        knifename_whole = 'C:\AlexZheng\knife/'+knifename
        img_knife = cv.imread(knifename_whole,1)
        imgs_list.append(img_knife)
        labels_string.append(1)

    # for oldman in os.listdir(r'C:\AlexZheng\oldman'):
    #
    #     oldman_whole = 'C:\AlexZheng\oldman/' + oldman
    #     img_oldman = cv.imread(oldman_whole,1)
    #     imgs_list.append(img_oldman)
    #     labels_string.append(2)


    labels = np.asarray(labels_string, dtype=int)
    imgs_array = np.asarray(imgs_list)

    return imgs_array, labels

#split the dataset for training
def split_data(imgs, labels):
    X_train, X_test, y_trian, y_test = train_test_split(imgs,labels,test_size=0.1,random_state=58)

    return X_train, X_test, y_trian, y_test

# #train the model
def trian_model(X_trian, y_train):

    # model = Sequential([
    #     Flatten(input_shape=(180,180,3)),
    #     Dense(20,activation='relu'),
    #     Dense(4, activation='relu'),
    #     Dense(2,activation='softmax')
    # ])

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_trian, y_train, epochs=5, batch_size=2)

    return model

#evaluate the model
def evaluate_model(model, X_test, y_test):

    # cv.namedWindow('img', cv.WINDOW_NORMAL)
    # for img in X_test:
    #     cv.imshow('img',img)
    #     cv.waitKey(0)

    y_pred = model.predict(X_test)
    _, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', accuracy)

def visualize(model,):

    # #get the emotion we need
    # happy_img = cv.imread(r'C:\AlexZheng\emotion\happy.PNG', 1)
    # angry_img = cv.imread(r'C:\AlexZheng\emotion\angry.png', 1)
    #
    # #get the image of BBQ soup
    # img_BBQsoup = cv.imread('C:\AlexZheng\imshow\9_000.png', 1)
    # img_list = []
    # img_list.append(img_BBQsoup)
    # img_array = np.asarray(img_list)
    #
    # # get the data of emotion we need
    # y_pred = model.predict(img_array)
    # y_pred_float = np.asarray(y_pred, dtype=float)
    #
    # #get the data of emotion
    # happy_app = np.asarray(happy_img * y_pred_float[0], dtype=int)
    # angry_app = np.asarray(angry_img * y_pred_float[0], dtype=int)
    #
    # emotion_blend = happy_app + angry_app
    #
    # fig, ax = plt.subplots()
    # im = ax.imshow(emotion_blend)
    #
    # ax.axis('off')
    # plt.show()

    BBQ_list = []

    for BBQ_soup in os.listdir(r'C:\AlexZheng\BBQsoap'):
        BBQ_whole = 'C:\AlexZheng\BBQsoap/' + BBQ_soup
        img_BBQ = cv.imread(BBQ_whole,1)
        BBQ_list.append(img_BBQ)

    BBQ_array = np.asarray(BBQ_list)

    y_pred = model.predict(BBQ_array)
    print(y_pred)

def save_model(model):
    tf.saved_model.save(model,"saved/1")


def main():
    imgs, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(imgs,labels)
    model = trian_model(X_train,y_train)
    #evaluate_model(model,X_test,y_test)
    visualize(model)
    #save_model(model)

if __name__ == "__main__":
    main()



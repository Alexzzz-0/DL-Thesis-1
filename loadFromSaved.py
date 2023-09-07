import tensorflow as tf
import os
import cv2 as cv
import numpy as np

from keras import models;

def load_from_saved():
    model = tf.saved_model.load("saved/1")
    return model

def test_with_img(model):

    test_list = []

    for test in os.listdir(r'C:\AlexZheng\BBQsoap'):
        test_whole = 'C:\AlexZheng\BBQsoap/' + test
        test_img = cv.imread(test_whole,1)
        test_list.append(test_img)

    test_array = np.asarray(test_list)

    for data in test_array:
        test_array_reshape = np.reshape(data,newshape=(1,180,180,3),order='C')
        y_pred = model(test_array)
        print(y_pred)

def main():
    model = load_from_saved()
    test_with_img(model)

if __name__ == "__main__":
    main()

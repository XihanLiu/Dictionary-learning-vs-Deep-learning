import os
import cv2
from numpy import linalg as LA
from sklearn.cluster import KMeans
import numpy as np
import torch
import matlab.engine
train_path = r'/Users/liuxihan/Desktop/compressed_sensing/Dictionary-learning-vs-Deep-learning'
dataset_path = r'/Users/liuxihan/Desktop/compressed_sensing/CroppedYale/'


def form_dictionary(train_path, dataset_path):
    # form 39 groups dictionary
    groups_num = 39
    samples_num = 10
    train_list = []
    dictionary = np.empty(shape=[32256, groups_num*samples_num])
    with open(os.path.join(train_path, 'train.txt'), 'r') as train:
        for lines in train:
            ind = lines.find('.')
            file_name = lines[0:ind]
            train_list.append(file_name)
            # file_name: returns list of file_name
        for group in range(groups_num+1):
            group_ind = str(group).rjust(2, '0')
            num = 0
            for sample in train_list:
                if sample[0:7] == "yaleB" + group_ind:
                    image = cv2.imread(os.path.join(
                        dataset_path, sample[0:7], sample+".pgm"))
                    height, width, _ = image.shape
                    resized_image = cv2.resize(
                        image[:, :, 1], (1, height*width))
                    np.append(dictionary, resized_image)

                    num += 1
                    if samples_num == num:
                        break
        #dictionary.shape = (32256,390)
        # print(dictionary.shape)
    return dictionary


def Omp(y, A, K):
    rows, cols = A.shape
    res = y
    indexs = []
    A_c = A.copy()

    for i in range(0, K):
        products = []
        for col in range(cols):
            products.append(np.dot(A[:, col].T, res))
        index = np.argmax(np.abs(products))
        indexs.append(index)
        inv = np.dot(A_c[:, indexs].T, A_c[:, indexs])
        theta = np.dot(np.dot(np.linalg.inv(inv), A_c[:, indexs].T), y)

        res = y-np.dot(A_c[:, indexs], theta)
    theta_final = np.zeros(cols,)
    theta_final[indexs] = theta

    return theta_final


def main():
    A = form_dictionary(train_path, dataset_path)
    eng = matlab.engine.start_matlab()
    test_list = []
    with open(os.path.join(train_path, 'test.txt'), 'r') as test:
        for lines in test:
            ind = lines.find('.')
            file_name = lines[0:ind]
            test_list.append(file_name)
            # file_name: returns list of file_name

    test_dataset = np.empty(shape=[32256, len(test_list)])
    for sample in test_list:
        image = cv2.imread(os.path.join(
            dataset_path, sample[0:7], sample+".pgm"))
        height, width, _ = image.shape
        # Vectorized Images
        resized_image = cv2.resize(image[:, :, 1], (1, height*width))
        np.append(test_dataset, resized_image)

        y = resized_image
        x_hat = Omp(resized_image, A, 15)
        print(x_hat)


main()

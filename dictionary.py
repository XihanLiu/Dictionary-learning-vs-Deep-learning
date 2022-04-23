import os
import cv2
from numpy import linalg as LA
from sklearn.cluster import KMeans
import numpy as np
import torch

train_path = r'/Users/liuxihan/Desktop/compressed sensing and sparse recovery/Dictionary-learning-vs-Deep-learning'
dataset_path = r'/Users/liuxihan/Desktop/compressed sensing and sparse recovery/CroppedYale/'


def form_dictionary():
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
    print(dictionary.shape)
    return dictionary


def OMP(y, A, K):
    x_ret = np.zeros((A, 2), 1)
    r = y
    s = []
    for i in range(K):
        cor = abs(np.transpose(A)*r)
        v, n = max(cor)
        s = s.union(n)
        x = torch.linalg.pinv(A(-1, s))*y
        old_norm_r = LA.norm(r)
        r = y - A[:, s]*x
        if LA.norm(r) < 1e-12 or LA.norm(r) >= old_norm_r:
            break
    x_ret[s] = x
    return x_ret


def main():
    A = form_dictionary()
    test_list = []
    with open(os.path.join(train_path, 'test.txt'), 'r') as test:
        for lines in test:
            ind = lines.find('.')
            file_name = lines[0:ind]
            test_list.append(file_name)
            # file_name: returns list of file_name

    test_dataset = np.empty(shape=[32256, len(test_list)])
    for sample in test_list:
        residual = np.zeros([1, 39])
        image = cv2.imread(os.path.join(
            dataset_path, sample[0:7], sample+".pgm"))
        height, width, _ = image.shape
        resized_image = cv2.resize(image[:, :, 1], (1, height*width))
        np.append(test_dataset, resized_image)
        x_hat = OMP(resized_image, A, 15)


main()

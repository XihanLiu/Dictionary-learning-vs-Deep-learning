from PIL import Image
import numpy as np
import os
from scipy.io import savemat

image = Image.open('test_dataset/yaleB01_5')
[m, n] = image.size

imagelist = os.listdir('test_dataset/')
rootdir = "test_dataset/"
testset = np.zeros((m * n, len(imagelist)))
testlable = np.zeros((1, len(imagelist)))
num = 0
for i in imagelist:
    img = Image.open(rootdir + i)
    mat = np.array(img)
    if mat.size == 307200:
        mat = mat[0:192, 0:168]
        mat = mat.reshape(m * n, 1)
    else:
        mat = np.array(img).reshape(m * n, 1)

    testlable[:, [num]] = int(i[5:7])
    testset[:, [num]] = mat
    num = num + 1

# print(testlable)
# print(testset)

image = Image.open('train_dataset/yaleB01_3')
[m, n] = image.size

imagelist = os.listdir('train_dataset/')
rootdir = "train_dataset/"
trainset = np.zeros((m * n, len(imagelist)))
trainlable = np.zeros((1, len(imagelist)))
num = 0
for i in imagelist:
    img = Image.open(rootdir + i)
    mat = np.array(img)
    if mat.size == 307200:
        mat = mat[0:192, 0:168]
        mat = mat.reshape(m * n, 1)
    else:
        mat = np.array(img).reshape(m * n, 1)

    trainlable[:, [num]] = int(i[5:7])
    trainset[:, [num]] = mat
    num = num + 1

file_name = 'CSdataset.mat'
savemat(file_name, {'testset': testset, 'testlable': testlable, 'trainset': trainset, 'trainlable': trainlable})


import os
import cv2
from numpy import linalg as LA
from sklearn.cluster import KMeans
import numpy as np
import torch
from traitlets.traitlets import directional_link

#train_path = r'/content/Dictionary-learning-vs-Deep-learning/'
#dataset_path = r'/content/CroppedYale/'
train_path = r'/Users/liuxihan/Desktop/compressed_sensing/Dictionary-learning-vs-Deep-learning'
dataset_path = r'/Users/liuxihan/Desktop/compressed_sensing/CroppedYale/'

vector_length = 32256 #width*height
groups_num = 39
samples_num = 10
def form_dictionary(train_path, dataset_path):
    # form 39 groups dictionary
    train_list =[]
    dictionary = np.array([], dtype=np.int64).reshape(vector_length,0)
    #dictionary=[]
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
                if sample[0:7] == "yaleB" + group_ind and "Ambient" not in sample:
                    
                    image = cv2.imread(os.path.join(dataset_path, sample[0:7], sample+".pgm"))
                    height, width, _ = image.shape
                    resized_image = cv2.resize(image[:, :, 1], (height*width,1)).T
                    dictionary = np.column_stack((dictionary, resized_image))
                    num += 1
                    if samples_num == num:
                        break
        
    train.close()   
    return dictionary


def Omp(y, A, K):
    cols = A.shape[1]
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
        res = y - np.dot(A_c[:, indexs], theta)

    theta_final = np.zeros(cols,)
    theta_final[indexs] = theta.reshape(len(indexs),)
    return theta_final

def main():
    #A.shape = (32256,380)
    train_dataset = form_dictionary(train_path, dataset_path)
    test_list = []
    label_list = []
    accuracy = []
    with open(os.path.join(train_path, 'test.txt'), 'r') as test:
        for lines in test:
            ind = lines.find('.')
            file_name = lines[0:ind]
            test_list.append(file_name)
            label = lines[ind+4:-1]
            label_list.append(label)
            # file_name: returns list of file_name
            
    test_list = test_list[1:380]
    label_list = label_list[1:380]
    
    #test_dataset.shape = (32256, 1187)
    test_dataset = np.empty(shape=[vector_length, len(test_list)])
    #sparse_coef.shape = (380,1187)
#     sparse_coef = np.zeros([A.shape[1],len(test_list)])
    predict = []
    
    for i in range(len(test_list)):
        with open(os.path.join(train_path, 'result.txt'), 'w') as result:
            sample = test_list[i]
            image = cv2.imread(os.path.join(dataset_path, sample[0:7], sample+".pgm"))
            height, width, _ = image.shape
            # Vectorized Images
            resized_image = cv2.resize(image[:, :, 1],(1,vector_length))
            np.append(test_dataset, resized_image)
            residual_list = []
            for j in range(groups_num):
                A = train_dataset[:,j*samples_num:(j+1)*samples_num]
                theta_final = Omp(resized_image, A, samples_num)
                residual = LA.norm(resized_image-np.dot(A,theta_final),2)
                residual_list.append(residual)
            min_residual = min(residual_list)
            print(min_residual)
            
            
            
            
            
            
            
#             y = resized_image
#             theta_final = Omp(resized_image, A, 15)
#             sparse_coef[:,i] = theta_final
#             residual_list = []
#             for j in range(groups_num):
#                 y_hat = np.dot(A[:,j*samples_num:(j+1)*samples_num],theta_final[j*samples_num:(j+1)*samples_num])
#                 residual = LA.norm(y-y_hat,2)
#                 residual_list.append(residual)
            
#             min_residual = min(residual_list)
#             print(min_residual)
#             predict_index = error_list.index(min_error)
#             result.write((sample+".pgm"+ ' ' + str(predict_index)))
#             print(sample+".pgm"+ ' ' + str(predict_index))   
    test.close()
    


main()



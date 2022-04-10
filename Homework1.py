import scipy.io as scio
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from scipy.optimize import linprog
import math
import operator
import random

def euclideanDistance(instance1, instance2, length):
     distance = 0
     for x in range(length):
         distance += pow((instance1[x]-instance2[x]), 2)
     return math.sqrt(distance)
def getNeighbors(trainingSet, testInstance, k):
     distances = []
     length = len(testInstance)-1
     for x in range(len(trainingSet)):
         dist = euclideanDistance(testInstance, trainingSet[x], length)
         distances.append((trainingSet[x], dist))   #get the distance from test_sample to train_sample
     distances.sort(key=operator.itemgetter(1))    #sort the distance of all the sample
     neighbors = []
     for x in range(k):   #get the k nearest sample
         neighbors.append(distances[x][0])
     return neighbors
def Omp(y, A, K):
    cols = A.shape[1]
    res = y
    indexs = []
    A_c = A.copy()

    # The recursion number is K
    for i in range(0, K):
        products = []
        for col in range(cols):
            products.append(np.dot(A[:, col].T, res))
        index = np.argmax(np.abs(products))
        indexs.append(index)
        inv = np.dot(A_c[:, indexs].T, A_c[:, indexs])  #

        theta = np.dot(np.dot(np.linalg.inv(inv), A_c[:, indexs].T), y)
        res = y - np.dot(A_c[:, indexs], theta)


    theta_final = np.zeros(cols, )
    theta_final[indexs] = theta
    return theta_final


data = scio.loadmat('.\YaleB_32x32.mat') #Dataset
fea,gnd = data['fea'],data['gnd'];
gnd = gnd.squeeze();
row,col = np.shape(fea)
class_num = int(max(gnd))
p_list = [1,2,3,5,10]
k_list = [2,3,5,10]
accuracy_matrix = np.zeros([len(p_list),len(k_list)])
#Spilt the data
#for q in range(len(k_list)):
for idx1 in range(50,60,10):
    accuracy = []
    for idx2 in range(len(k_list)):
        val_num = 20
        train_dataset = np.zeros([1,col])
        train_label = []
        test_dataset = np.zeros([1,col])
        test_label = []
        #valid_dataset = np.zeros([1,col])
        #valid_label = []
        for n in range(1,max(gnd)+1):
            idx = np.where(gnd==n)
            temp1 = fea[idx,:]
            temp1 = temp1.squeeze();
            np.random.shuffle(temp1)
            train = temp1[:idx1]
            test = temp1[idx1:]
            #valid = temp1[m:m+val_num]
            #train = temp1[m+val_num:]
            p,_ = np.shape(test) #Remaining number of train data
            train_dataset = np.concatenate((train_dataset,train),axis = 0)
            for i in range(idx1):
                train_label.append(n);
            test_dataset = np.concatenate((test_dataset,test),axis=0)
            for i in range(p):
                test_label.append(n);
            # valid_dataset = np.concatenate((valid_dataset,valid),axis=0)
            # for i in range(val_num):
            #     valid_label.append(n)
        train_dataset = np.delete(train_dataset,0,axis=0)
        test_dataset = np.delete(test_dataset,0,axis=0)
        # valid_dataset = np.delete(valid_dataset,0,axis = 0)
        #compute hog features
    # train_hog = np.zeros([train_dataset.shape[0],train_dataset.shape[1]])
    # test_hog = np.zeros([test_dataset.shape[0],test_dataset.shape[1]])
    # for i in range(train_dataset.shape[0]):
    #     feature = train_dataset[i,:].reshape(32,32)
    #     _,hog_feature = hog(feature,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1),visualize=True)
    #     hog_feature = hog_feature.flatten()
    #     train_hog[i,:] =hog_feature
    # for i in range(test_dataset.shape[0]):
    #     feature = test_dataset[i,:].reshape(32,32)
    #     _,hog_feature = hog(feature,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
    #     hog_feature = hog_feature.flatten()
    #     test_hog[i,:] =hog_feature
    # #compute LBP features
    # train_lbp = np.zeros([train_dataset.shape[0], train_dataset.shape[1]])
    # test_lbp = np.zeros([test_dataset.shape[0], test_dataset.shape[1]])
    # for i in range(train_dataset.shape[0]):
    #     feature = train_dataset[i,:].reshape(32,32)
    #     lbp_feature = local_binary_pattern(feature, 12, 3,method = 'uniform')
    #     lbp_feature = lbp_feature.flatten()
    #     train_lbp[i,:] =lbp_feature
    # for i in range(test_dataset.shape[0]):
    #     feature = test_dataset[i,:].reshape(32,32)
    #     lbp_feature = local_binary_pattern(feature, 12,3,method='uniform')
    #     lbp_feature = lbp_feature.flatten()
    #     test_lbp[i,:] =lbp_feature
    # train_label1 = np.array(train_label)
    # train_label1 = train_label1[:,np.newaxis]
    # test_label1 = np.array(test_label)
    # test_label1 = test_label1[:, np.newaxis]
#LDA model
#     model = LDA()
#     model.fit(train_dataset,train_label)
#     predict = model.predict(test_dataset)
#     accuracy.append(accuracy_score(test_label, predict))
# #SRC model
        train_dataset = np.transpose(train_dataset)
        test_dataset = np.transpose(test_dataset)
        m=10
        predict = []
        num = test_dataset.shape[1]
        sparse_coef = np.zeros([train_dataset.shape[1],num])
        for i in range(test_dataset.shape[1]):
            a = Omp(test_dataset[:,i],train_dataset,K=100)
            sparse_coef[:,i]=a
            error_list = []
            for j in range(class_num):
                y_hat = np.dot(train_dataset[:,j*m:(j+1)*m],a[j*m:(j+1)*m])
                error = sum((test_dataset[:, i]-y_hat)**2)
                error_list.append(error)
            min_error = min(error_list)
            predict_index = error_list.index(min_error)
            predict.append(predict_index+1)
        accuracy.append(accuracy_score(test_label, predict))



# PCA model
#     pca = PCA(n_components=200).fit(train_dataset)
#     mean = np.mean(train_dataset,axis=0)
#     project_train = np.dot(train_dataset-mean,np.transpose(pca.components_))
#     project_test =  np.dot(test_dataset-mean,np.transpose(pca.components_))
#     k=1;
#     model = KNeighborsClassifier(n_neighbors=k,p=2);
#     model.fit(project_train,train_label)
#     predict = model.predict(project_test)
#     accuracy.append(accuracy_score(test_label,predict))

#KNN Model
        k = k_list[idx2];
        model = KNeighborsClassifier(n_neighbors=k);#p_list[idx1]);
        model.fit(train_dataset,train_label)
        predict = model.predict(test_dataset)
        result = accuracy_score(test_label, predict)
        idx_list = []
        for i in range(len(test_label)):
            if test_label[i]!=predict[i]:
                idx_list.append(i)
#choose a mislabeled sample randomly
        mis_labeled_idx = random.sample(idx_list,1)
        mis_labeled_sample = test_dataset[mis_labeled_idx]
        neighbors = getNeighbors(train_dataset, test_dataset[3], k)
        plt.figure()
        row_num = np.ceil((k+1) / 3)
        col_num = 3
        plt.subplot(row_num, col_num, 1)
        data = mis_labeled_sample.reshape(32,32)
        plt.imshow(data)
        plt.title('Misclassified sample')
        for i in range(k):
            plt.subplot(row_num,col_num,i+2)
            data = neighbors[i].reshape(32,32)
            plt.imshow(data)
        plt.show()
        result = accuracy_score(test_label,predict)
        # result = (round(100 * (1 - result), 3))
        #         accuracy.append(accuracy_score(test_label,predict))
        #     accuracy = [(round(100 * (1 - i), 3)) for i in accuracy]
        #     accuracy_matrix[idx1, :] = accuracy
#     SVM Model
#         model = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovo', kernel='rbf')
#         model.fit(train_dataset,train_label1)
#         predict = model.predict(test_dataset)
#         accuracy.append(accuracy_score(test_label1,predict))
# accuracy = [(round(100*(1-i),3)) for i in accuracy]
    #accuracy_matrix[q,:]=accuracy

#plot the curve against p
# plt.figure()
# for i in range(accuracy_matrix.shape[0]):
#     plt.plot(k_list,accuracy_matrix[i,:],'-o')
# plt.legend(['p=1','p=2','p=3','p=5','p=10'])
# plt.xlabel('k')
# plt.ylabel('Classification error rate')
# plt.title('Validation dataset performance')
# plt.show()

# plt.figure();
# plt.plot(range(10,60,10),accuracy,'-ro')
# #plt.plot(p_list,accuracy,'-ro')
# plt.xlabel('Number of Trainings Samples')
# plt.ylabel('Classification Error Rate')
# for a,b in zip(range(10,60,10),accuracy):
#     plt.text(a, b,b, ha='center', va='bottom', fontsize=10)
# plt.title('Part3 Sparse Representation K=100')
# plt.show()



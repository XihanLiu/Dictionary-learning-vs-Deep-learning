clear all; 
close all; 
clc;
load('CroppedYale_96_84_2414_subset.mat');
%randomly sperate training set and testing set (half and half)
[m,n] = size(facecls);
random_set = randi(2,2414,1);
training_set= [facecls(find(random_set==1)) find(random_set==1)];
testing_set = [facecls(find(random_set==2)) find(random_set==2)];
q = randperm(1200);
testing_set = testing_set(q(1:300),:);
class = facecls(end);
%form the index dictionary
training_sample = 25;
dict_index = zeros(training_sample,class);
accuracy = 0;
for i = 1:class
    dict_index(:,i) = training_set(find(training_set(:,1)==i,1):find(training_set(:,1)==i,1)+(training_sample-1),2);
end
%give a test image
% y = double(faces(testing_set(1,2),:,:));
% give test set
[len,~] = size(testing_set);
for test_i = 1:len
    y = double(faces(testing_set(test_i,2),:,:));
    y = y(:);
    % define non-zero entries in sparse code x
%     theta = training_sample;
    % should be the value in dict
    residual = zeros(1,class);
%     class_matrix = [];
%     A = [];
    for i = 1:class
        class_matrix = [];
        A = [];
        for j = 1:training_sample
            tmp_matrix_index = dict_index(j,i);
            tmp_matrix = reshape(faces(tmp_matrix_index,:,:),96,84);
            class_matrix = double([class_matrix tmp_matrix(:)]);
        end
        A = class_matrix;
%         x_hat = OMP(y,A,training_sample);
        [x_hat, nnIt] = FISTA(A,y);
        residual(i) = norm(y-A*x_hat);
    end
    [~,testing_set(test_i,3)]= min(residual);    
    if testing_set(test_i,1)==testing_set(test_i,3)
        accuracy = accuracy+1;
    end
end
accuracy_rate = accuracy/len;



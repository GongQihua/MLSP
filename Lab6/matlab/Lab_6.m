%% Lab 6: MNIST Digit Classification with kNN and SVMs
clc; 
clear all; 
close all;
%% Dataset
% Load the train and test data into matrices using 'loadMNISTImages' and 
% 'loadMNISTLabels'

% Your code here
train_imgs = loadMNISTImages('../mnist/train-images-idx3-ubyte')'; 
train_label = loadMNISTLabels('../mnist/train-labels-idx1-ubyte'); 
test_imgs = loadMNISTImages('../mnist/t10k-images-idx3-ubyte')'; 
test_label = loadMNISTLabels('../mnist/t10k-labels-idx1-ubyte');
train_imgs = train_imgs';
test_imgs = test_imgs';
%% PCA
% Find the PCA bases and project the train and test data using the PCA bases

% Your code here
num_PCA = 65; %change the num of PCA here
m = size(train_imgs, 1);
n = size(train_imgs, 2);
meanface = mean(train_imgs,1);%get meanface
mat = double(train_imgs);
for i = 1:m
    mat(i,:) = train_imgs(i,:) - meanface;
end
C = mat' * mat;
[eig_train,~] = eigs(C, num_PCA); %get eig
trainproj = train_imgs * eig_train;%projection matrix
testproj = test_imgs * eig_train;
%% KNN
% Implement the K-nearest neighbor algorithm to classify test data into the
% 10 digit classes. 

% Your code here
num_K = 3;% change the number of KNN here 
train_scale = size(trainproj);% build the new test and train classify set
test_scale = size(testproj);
test_classify = zeros(test_scale(1),1);
disp('running the KNN model');
tic;
for i = 1:test_scale(1)
    test_point = testproj(i,:);
    dist = zeros(train_scale(1),1);% distance matrix
    for j=1:train_scale(1)
        train_point = trainproj(j, :); % calculate the distance between test and train
        tmp = test_point - train_point;
	    dist(j) = sqrt(sum(tmp.*tmp));
    end
    
    newdist = sort(dist);
    num = zeros(10, 1);
    for k=1:num_K %K-nearest find
       idx = find(dist == newdist(k));
       num(train_label(idx) + 1) = num(train_label(idx) + 1) + 1;
    end
    
    max_Idx = 0;
    max_Num = -1;
    for k = 1:10
      if(num(k) > max_Num) %find max num
         max_Idx = k;
         max_Num = num(k);
      end
    end 
    test_classify(i) = max_Idx - 1;
end
toc;
correct = sum(test_label == test_classify);
accuracy_knn = correct / test_scale(1); %print accuracy
%% SVM
% Use MATLAB's SVM to classify test data into the 10 digit classes.

% Your code here
disp('running the SVM model');
tic;
t = templateSVM('KernelFunction','linear', 'BoxConstraint',0.75);%load function
SVM = fitcecoc(trainproj, train_label,'Learners',t);
result = predict(SVM,testproj);% calculate acc
result = result.';
acc = 0.;
for i = 1:10000
    if result(i) == test_label(i)
        acc = acc+1;
    end
end
accuracy_svm = acc/10000;
toc;
%% Remember to write a report. Submit your .m files and report as one zip file
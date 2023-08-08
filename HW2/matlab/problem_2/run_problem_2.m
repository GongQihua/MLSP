%% Driver File for Problem 2: Boosting-based Face Detector
% Implement an Adaboost Classifier to classify between face images and 
% non-face images
% Author Name : Qihua Gong

%% Your Script Starts Here
% You can create as many auxilliary scripts as you want
% As long as we can run this script to get all the plots and classification
% accuracies we require
clc;
clear;
close all;
%% Compute the first K Eigenfaces.
filepath = dir('../../data/lfw_1000/*.pgm');
data = zeros(64*64, size(filepath,1)); 
mat = zeros(64*64, 1); 

for i=1:length(filepath)
    im = imread([filepath(i).folder '/' filepath(i).name]);
    img = double(reshape(im, [64*64,1]));
    data(:,i) = img;
    mat = mat + img;
end
mat = mat/size(filepath,1);%lfw matrix

for i=1:length(filepath)
    data(:,i) = data(:,i) - mat;
end
[U, S, V] = svd(data);%compute eig
K = 30;% change the K here

%% load training data and Project the train Images.
path = dir('../../data/boosting/train/face/*.pgm');
trainFaces = zeros(19*19, size(path,1));
for i=1:length(path)
    im = imread([path(i).folder '/' path(i).name]);
    img = double(reshape(im, [19*19,1]));
    trainFaces(:,i) = img;
end

path = dir('../../data/boosting/train/non-face/*.pgm');
trainNonFaces = zeros(19*19, size(path,1));
for i=1:length(path)
    im = imread([path(i).folder '/' path(i).name]);
    img = double(reshape(im, [19*19,1]));
    trainNonFaces(:,i) = img;
end

path = dir('../../data/boosting/test/face/*.pgm');
testFaces = zeros(19*19, size(path,1));
for i=1:length(path)
    im = imread([path(i).folder '/' path(i).name]);
    img = double(reshape(im, [19*19,1]));
    testFaces(:,i) = img;
end

path = dir('../../data/boosting/test/non-face/*.pgm');
testNonFaces = zeros(19*19, size(path,1));
for i=1:length(path)
    im = imread([path(i).folder '/' path(i).name]);
    img = double(reshape(im, [19*19,1]));
    testNonFaces(:,i) = img;
end

%% Project the Face and Non-Face Images.
eigfaces = U(:,1:K);
eig_rfaces = zeros(19*19, K);
for i=1:K
    eigface = eigfaces(:,i);
    newface = double(reshape(imresize(reshape(eigface, [64,64]), [19,19]), [19*19,1]));
    eig_rfaces(:,i) = newface;
end

faceweight = trainFaces'*eig_rfaces;%train feature
nonfaceweight = trainNonFaces'*eig_rfaces;
train = [ones(2429,1); -1*ones(4548,1)];
train_weight = [faceweight; nonfaceweight];

face_test = testFaces'*eig_rfaces;%test feature
nonface_test = testNonFaces'*eig_rfaces;
test = [ones(size(face_test,1),1); -1*ones(size(nonface_test,1),1)];
test_weight = [face_test; nonface_test];

%% Classify between Face and Non-Face.
% train an Adaboost classifier to classify between Face and Non-Face images.
num_iters = 10; % number of iterations 
weights = (1/size(train_weight,1))*ones(size(train_weight,1),1); % calculate weight 
alphas = zeros(num_iters, 1);
Adaboost = {};
dims = {};

for iter=1:num_iters
    min_dis = inf;%find min distance
    for dim=1:K
        cmat = fitctree(train_weight(:,dim), train, 'minparent',size(train,1),'prune','off','mergeleaves','off', 'Weights',weights);
        dis = loss(cmat, train_weight(:,dim), train, 'Weights', weights);
        if dis < min_dis
            min_dis = dis;
            best_dis = cmat;
            dimension = dim;
        end
    end
    dims{iter} = dimension;
    Adaboost{iter} = best_dis;
    alpha = 0.5*log((1-min_dis)/min_dis);
    alphas(iter) = alpha;
    label = predict(best_dis, train_weight(:,dim));
    for i=1:size(train_weight,1)% new weights
        weights(i) = weights(i)*exp(-alpha*train(i)*label(i));
    end 
    weights = weights/sum(weights); %norm
end

%% Report the overall classification accuracy
acc = zeros(size(test,1),1);
for iter=1:num_iters
    acc = acc + alphas(iter)*predict(Adaboost{iter}, test_weight(:,dims{iter}));
end

predict = sign(acc);
accuracy = sum(test == predict) / size(test,1);
disp(" The overall classification accuracy is " + accuracy);
% for k = 10 the overall classification accuracy is 0.80082
%for k = 20 the overall classification accuracy is 0.71036
%for k = 30 the overall classification accuracy is 0.69414
%With the k increase, the accuracy decrease
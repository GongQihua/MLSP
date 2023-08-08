%% Driver File for Problem 3: Gender Detector
% implement a gender detection system using the PCA dimensions from images
% Author Name : Qihua Gong



%% Your Script Starts Here
% You can create as many auxilliary scripts as you want
% As long as we can run this script to get all the plots and classification
% accuracies we require
clc;
clear all;
close all;
%% Load all data.
path =strcat( '../../data/lfw_genders/male/train/');
filenames  = dir([path '*.jpg']); 
matX = []; 
flatTrain = (size(imread([path 'Adel_Al-Jubeir_0001.jpg']), 1)^2);
for i = 1 : length(filenames)
    datapath = [path filenames(i).name];
    img = double(imread(datapath));
    matX(:,i) = reshape(img, flatTrain, 1);
end
male = matX/max(matX(:)); % and made the loading data into a matrix

path =strcat( '../../data/lfw_genders/female/train/');
filenames  = dir([path '*.jpg']); 
matX = []; 
flatTrain = (size(imread([path 'Ai_Sugiyama_0001.jpg']), 1)^2);
for i = 1 : length(filenames)
    datapath = [path filenames(i).name];
    img = double(imread(datapath));
    matX(:,i) = reshape(img, flatTrain, 1);
end
female = matX/max(matX(:));

path =strcat( '../../data/lfw_genders/female/test/');
filenames  = dir([path '*.jpg']); 
matX = []; 
flatTrain = (size(imread([path 'AJ_Cook_0001.jpg']), 1)^2);
for i = 1 : length(filenames)
    datapath = [path filenames(i).name];
    img = double(imread(datapath));
    matX(:,i) = reshape(img, flatTrain, 1);
end
test_female = matX/max(matX(:));

path =strcat( '../../data/lfw_genders/male/test/');
filenames  = dir([path '*.jpg']); 
matX = []; 
flatTrain = (size(imread([path 'Aaron_Pena_0001.jpg']), 1)^2);
for i = 1 : length(filenames)
    datapath = [path filenames(i).name];
    img = double(imread(datapath));
    matX(:,i) = reshape(img, flatTrain, 1);
end
test_male = matX/max(matX(:));
all_train = [female male];

%% Apply PCA and Find PCA dimensions
center = all_train - mean(all_train, 'all');
[U,S,V] = svd(center, 0); % get the eig values and eig vector
K_cells = {};
count = 1;
for K = [50, 100, 200, 300]
    eigFaces = U(:,1:K); %find he K best eig vectors and save it
    K_cells{1,count} = eigFaces;
    eigVals = diag(S); 
    eigVals = eigVals(1:K);
    K_cells{2,count} = eigVals;
    count = count+1; 
end

%% select K
figure(1);
for fig = 1 : 4 %plot the eig values
    subplot(2,2,fig);
    plot(K_cells{2,fig});
    ylim([0 500]);
    title('eig values');
end
% I think the K=50 captures sufficient information, it looks has already
% capture the most information in the eigenvectors but when k = 300, it
% undoubtly capture the most information

%% Find the average Face.
avg_male = [];
avg_female = [];
for row = 1 : 62500 % calculate the average
    avg_male   = [avg_male mean(male(row,:))];
    avg_female = [avg_female mean(female(row,:))];
end
figure(2);
subplot(1,2,1);
imshow( reshape( avg_male, [250 250]), []);
title('average male face');
subplot(1,2,2);
imshow( reshape( avg_female, [250 250]), []);
title('average female face');

%% Detect the gender
%Project your average male and female face on your K PCA dimensions.
avgs = {};
avg_weights = {};
count = 1;
for K = [50, 100, 200, 300] % calculate the avg data and weights
    E = U(:,1:K);
    X = reshape(avg_male, [62500 1]);
    w = E' * X;
    male_proj = E*w;
    avgs{1,count} = male_proj;
    avg_weights{1,count} = w;
    
    E = U(:,1:K);
    X = reshape(avg_female, [62500 1]);
    w = E' * X;
    female_proj = E*w;
    avgs{2,count} = female_proj;
    avg_weights{2,count} = w;
    
    count = count + 1;
end

% Project all your testing images on your K PCA dimensions.
all_test = [test_male, test_female];
testing = {};
c1 = 1;
testWeights = {};
for K = [50, 100, 200, 300] % calculate the weights of each testing image
    E = U(:,1:K);
    c2 = 1;
    for face = 1 : size(all_test, 2)
        X = all_test(:,face);
        w = E' * X;
        proj = E*w;
        testing{c1,c2} = proj;
        testWeights{c1,c2} = w;
        c2 = c2 + 1;
    end
    c1 = c1 + 1;
end

% Classify each testing image as male or female
test_dist = ones(size(testing));
for K = 1 : 4
    MaleavgW  = avg_weights{1,K}; 
    FemaleavgW  = avg_weights{2,K};  
    for testWeight = 1 : length(testing)%compare avg train weight and test weight
        test_weight = testWeights{K,testWeight};
        maleDist   = norm(test_weight  - MaleavgW);
        femaleDist = norm(test_weight  - FemaleavgW);

        if maleDist < femaleDist %define the male or female, 1 is male, 0 is female
            test_dist(K,testWeight) = 1;
        else
            test_dist(K,testWeight) = 0;
        end
        
    end    
end

K2 = [50, 100, 200, 300];
for K = 1:4 %% Calculate the accuracy in each K value
    tic
    accmale = sum(test_dist(K,1:1000) == 1) / 1000;
    accfemale = sum(test_dist(K,1000:end) == 0) / 1000;
    disp("K=" + K2(K) + ":");
    disp(" -correct male accuracy of " + accmale);
    disp(" -correct female accuracy of " + accfemale);
    toc
    disp(" ");
end
disp("detect the all gender");
disp(" ");
% First for a conclusion of my result
%K=50:
%-correct male accuracy of 0.521
%-correct female accuracy of 0.601
%time 0.010432 秒。
 
%K=100:
%-correct male accuracy of 0.521
%-correct female accuracy of 0.604
%time 0.001279 秒。
 
%K=200:
%-correct male accuracy of 0.521
%-correct female accuracy of 0.603
%time 0.001001 秒。
 
%K=300:
%-correct male accuracy of 0.521
%-correct female accuracy of 0.604
%time 0.000124 秒。
% We can see that by the K increase, the accuracy of male doesn't change a
% lot but the accuracy of female have a little increase, the runing time
% will decrease a bit. Actually the accuracy not changes a lot.

%% Detect the all gender:
% Project all the training images into the K PCA dimensions, and all the testing images into the K PCA dimensions.
%nearly the same process as it above
trainingmale = {}; 
trainmaleWeights = {}; 
c1 = 1;
for K = [50, 100, 200, 300] %calculate the all male train matrix and female train matrix weight
    E = U(:,1:K);
    c2 = 1;
    for trainFace = 1:size(male,2)
        X = male(:,1934);
        w = E' * X;
        proj = E*w;
        trainingmale{c1,c2} = proj;
        trainmaleWeights{c1,c2} = w;
        c2 = c2 + 1;
    end
    c1 = c1 + 1;
end

trainingfemale = {}; 
trainfemaleWeights = {}; 
c1 = 1;
for K = [50, 100, 200, 300]
    E = U(:,1:K);
    c2 = 1;
    for trainFace = 1:size(female,2)
        X = female(:,1934);
        w = E' * X;
        proj = E*w;
        trainingfemale{c1,c2} = proj;
        trainfemaleWeights{c1,c2} = w;
        c2 = c2 + 1;
    end
    c1 = c1 + 1;
end

decision = []; 
dist_male = [];
dist_female = [];
for k = 1:4
    for testWeight = 1 : 1934 % Calculate the average distance
        test_all = testWeights{k,testWeight};
        train_male = trainmaleWeights{k,testWeight};
        train_female = trainfemaleWeights{k,testWeight}; 
        dist_male = norm(test_all - train_male,1);
        dist_female = norm(test_all - train_female,1);

        if dist_male < dist_female
            decision(k,testWeight) = 1;
        else
            decision(k,testWeight) = 0;
        end

    end
end

K2 = [50, 100, 200, 300];
for K = 1:4 % output the accuracy
    tic
    maleAccuracy = sum(decision(K,1:1000)==1)/1000;
    femaleAccuracy = sum(decision(K,1000:end)==0)/1000;
    disp("K=" + K2(K) + ":");
    disp(" -correct all male accuracy of " + maleAccuracy);
    disp(" -correct all female accuracy of " + femaleAccuracy);
    toc
    disp(" ");
end
% In the all result, we can see the result more obvious. As the K increase, the
% the accuracy decrease and the time also decrease. Therefore, it is more 
% important to choose a suitable K to balance the accuracy and time during 
% the training process.
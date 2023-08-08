clc;
clear all;
close all;
%% load data and Normalizing the length of all I-vectors

train_path = "../Data/Train/Class_";
dev_path = "../Data/Dev/Class_";
eval_path = "../Data/Eval/Class_";

train = loaddata(train_path);
dev = loaddata(dev_path);
eval = loaddata(eval_path);

%% count and normalize Ivector
L = 24;
A5 = train{1};
count_Ivector(1,1) = size(train{1},2);

for i = 2:L
    A5 = [A5 train{i}];
    count_Ivector(i,1) = size(train{i},2);
end  

for i = 1:L %calculate mean
    w2(:,i) = mean(train{i},2);
end

%% shared corvariance

for i = 1:L
    sigma{i} = zeros(size(A5,1), size(A5,1));
    for j = 1:size(train{i},2)
        sigma{i} = sigma{i} + (train{i}(:,j) - w2(:,i))*(train{i}(:,j) - w2(:,i))'; 
    end
    sigma{i} = (1/size(train{i},2))*sigma{i};
end

sigma_training = zeros(size(A5,1), size(A5,1));
for i = 1:L
    sigma_training = sigma_training + (count_Ivector(i,1)/size(A5,2))*sigma{i};
end


%% classify and calculate the accuracy

inverse_sigma = inv(sigma_training);
for i = 1:L
    for j = 1:size(dev{i},2)
        for k = 1:L
            dev_matrix{i}(k,j) = (dev{i}(:,j)-w2(:,k))'*inverse_sigma*(dev{i}(:,j)-w2(:,k));
        end
    end 
end

count_dev5 = 0;
for i = 1:L
    for j = 1:size(dev_matrix{i},2)
        [~, idx1{i}(j,1)] = min(dev_matrix{i}(:,j));
        if idx1{i}(j,1)==i
            count_dev5 = count_dev5 + 1;
        end
    end
end
acc_dev5 = count_dev5/2400; 

for i = 1:L
    for j = 1:size(eval{i},2)
        for k = 1:L
            eval_matrix{i}(k,j) = (eval{i}(:,j)-w2(:,k))'*inverse_sigma*(eval{i}(:,j)-w2(:,k));
        end
    end 
end

count_eval5 = 0;
for i = 1:L
    for j = 1:size(eval_matrix{i},2)
        [~, idx2{i}(j,1)] = min(eval_matrix{i}(:,j));
        if idx2{i}(j,1)==i
            count_eval5 = count_eval5 + 1;
        end
    end
end
acc_eval5 = count_eval5/2400; % 0.6504

disp('The test accuracy for dev:');disp(acc_dev5);
disp('The test accuracy for eval:');disp(acc_eval5);
clc;
clear all;
close all;

%% Data loading and Normalizing 

train_path = "../Data/Train/Class_";
dev_path = "../Data/Dev/Class_";
eval_path = "../Data/Eval/Class_";

train = loaddata(train_path);%loading and Normalizing
dev = loaddata(dev_path);
eval = loaddata(eval_path);

%% LDA training
% same like doing in the lab4
L = 24;
total = 0;
for i = 1:L
    total = total + size(train{i},2);
    col(:,i) = sum(train{i},2);     
end    
w1 = sum(col,2)./total;

Sb = zeros(size(w1,1), size(w1,1));
for i = 1:L %calculate sb
    w2(:,i) = mean(train{i},2);
    dis(:,i) = w2(:,i) - w1; %distance
    Sb = Sb + size(train{i},2)*dis(:,i)*dis(:,i)';
    
end    

Sw = zeros(size(w1,1), size(w1,1));
for i = 1:L %calculate sw
    for j = 1:size(train{i},2)
        Sw = Sw + (train{i}(:,j) - w2(:,i))*(train{i}(:,j) - w2(:,i))';   
    end
end

[V, D] = eigs(Sb, Sw, L-1);% use eig to calculate LDA

%% Classifier training

for i = 1:L  %projection and normalizing
    for j = 1:size(train{i},2)
        w_train{i}(:,j) = (V'*train{i}(:,j))/norm(V'*train{i}(:,j));   
    end
end    

for i = 1:L  %calculate the mean
    m(:,i) = mean(w_train{i},2)/norm(mean(w_train{i},2));
end   

%% Classifier testing

for i = 1:L %same as above, project the dev and eval
    for j = 1:size(dev{i},2)
        w_dev{i}(:,j) = (V'*dev{i}(:,j))/norm(V'*dev{i}(:,j));   
    end
end

for i = 1:L  
    for j = 1:size(eval{i},2)
        w_eva{i}(:,j) = (V'*eval{i}(:,j))/norm(V'*eval{i}(:,j));   
    end
end   

for i = 1:L %calculate the dot product
    score_deve{i} = w_dev{i}'*m;
end

for i = 1:L 
    score_eval{i} = w_eva{i}'*m;
end

count_dev = 0; %count dev total correct
for i = 1:L 
    [~,index]=max(score_deve{i},[],2);
    for j = 1:size(index,1)
        if index(j,1) == i
            count_dev = count_dev + 1;
        end
    end
end

count_eva = 0; %count eval total correct
for i = 1:L 
    [~,index]=max(score_eval{i},[],2); 
    for j = 1:size(index,1)
        if index(j,1) == i
            count_eva = count_eva + 1;
        end
    end
end

acc_dev = count_dev/2400; %get the accuracy
acc_eval = count_eva/2400; 
disp('The test accuracy for dev:');disp(acc_dev);
disp('The test accuracy for eval:');disp(acc_eval);
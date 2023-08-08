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

%% Use the after projecting first all of them by LDA

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
    dis(:,i) = w2(:,i) - w1;
    Sb = Sb + size(train{i},2)*dis(:,i)*dis(:,i)';   
end    

Sw = zeros(size(w1,1), size(w1,1));
for i = 1:L %calculate sw
    for j = 1:size(train{i},2)
        Sw = Sw + (train{i}(:,j) - w2(:,i))*(train{i}(:,j) - w2(:,i))';     
    end
end

[V, D] = eigs(Sb, Sw, L-1); % use eig to calculate LDA

%% Classifier training

for i = 1:L  %project and normalize
    for j = 1:size(train{i},2)
        w_train{i}(:,j) = (V'*train{i}(:,j))/norm(V'*train{i}(:,j));   
    end
end 

%% Classifier testing

for i = 1:L  %project the dev and eval
    for j = 1:size(dev{i},2)
        w_dev{i}(:,j) = (V'*dev{i}(:,j))/norm(V'*dev{i}(:,j));   
    end
end


for i = 1:L   
    for j = 1:size(eval{i},2)
        w_eval{i}(:,j) = (V'*eval{i}(:,j))/norm(V'*eval{i}(:,j));   
    end
end   

K = 55;
for i = 1:L %calculate kmeans
    if size(w_train{i},2) ~= K
        [idx4{i}, training_kmeans4{i}] =  kmeans(w_train{i}', K);
    else    
        training_kmeans4{i} = w_train{i}';
        temp = 1:K;
        idx4{i} = temp';
    end
    training_kmeans4{i} = training_kmeans4{i}';
end  

A4 = training_kmeans4{1}; %take kmeans into part2
count_Ivector(1,1) = size(training_kmeans4{1},2);

for i = 2:L
    A4 = [A4 training_kmeans4{i}];
    count_Ivector(i,1) = size(training_kmeans4{i},2);
end  

for i = 1:L %calculate dev alpha
    for j = 1:size(w_dev{i},2)
        temp = lasso(A4,w_dev{i}(:,j),'NumLambda',3);
        alpha_deve4{(i-1)*100+j}(:,1) = temp(:,1);
    end
end

for j = 1:2400 %final cell
    count_alpha = 0;
    temp_alpha = alpha_deve4{j};
    for k = 1:L
        temp = zeros(size(A4,2),1);
        temp((count_alpha+1):(count_alpha+count_Ivector(k,1))) = temp_alpha((count_alpha+1):(count_alpha+count_Ivector(k,1)));
        count_alpha = count_alpha + count_Ivector(k,1);
        cell_deve4{j}(:,k) = temp(:,1);
    end
end

for i = 1:L %calculate eval alpha
    for j = 1:size(w_eval{i},2)
        temp = lasso(A4,w_eval{i}(:,j),'NumLambda',3);
        alpha_eval4{(i-1)*100+j}(:,1) = temp(:,1);
    end
end

for j = 1:2400 %final cell
    count_alpha = 0;
    temp_alpha = alpha_eval4{j};
    for k = 1:L
        temp = zeros(size(A4,2),1);
        temp((count_alpha+1):(count_alpha+count_Ivector(k,1))) = temp_alpha((count_alpha+1):(count_alpha+count_Ivector(k,1)));
        count_alpha = count_alpha + count_Ivector(k,1);
        cell_eval4{j}(:,k) = temp(:,1);
    end
end

%% Compute the residuals
%same as formal part
for i = 1:L
    for j = 1:size(w_dev{i},2)
        dis_deve4{(i-1)*100+j} = w_dev{i}(:,j)-A4*cell_deve4{(i-1)*100+j};
        for k = 1:L
            res_deve4{(i-1)*100+j}(k,1) = norm(dis_deve4{(i-1)*100+j}(:,k));
        end        
    end
end

for i = 1:L
    for j = 1:size(w_eval{i},2)
        dis_eval4{(i-1)*100+j} = w_eval{i}(:,j)-A4*cell_eval4{(i-1)*100+j};
        for k = 1:L
            res_eval4{(i-1)*100+j}(k,1) = norm(dis_eval4{(i-1)*100+j}(:,k));
        end       
    end
end

%% identity and classify
count_deve4 = 0;
for i = 1:2400
    [~, idx1(i,1)] = min(res_deve4{i});
    if idivide(int32(i),int32(100),'ceil') == idx1(i,1)
        count_deve4 = count_deve4 + 1;
    end

end
acc_dev4 = count_deve4/2400; 

count_eval4 = 0;
for i = 1:2400
    [~, idx2(i,1)] = min(res_eval4{i});
    if idivide(int32(i),int32(100),'ceil') == idx2(i,1)
        count_eval4 = count_eval4 + 1;
    end

end
acc_eval4 = count_eval4/2400;

disp('The test accuracy for dev:');disp(acc_dev4);
disp('The test accuracy for eval:');disp(acc_eval4);
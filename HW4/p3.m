clc;
clear all;
close all;
%% data loading and Normalizing

train_path = "../Data/Train/Class_";
dev_path = "../Data/Dev/Class_";
eval_path = "../Data/Eval/Class_";

train = loaddata(train_path);
dev = loaddata(dev_path);
eval = loaddata(eval_path);

%% LDA training
% Estimate the between class covariance

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

[V, D] = eigs(Sb, Sw, L-1);% use eig to calculate LDA

%% Classifier training

for i = 1:L  %project and normalize
    for j = 1:size(train{i},2)
        w_train{i}(:,j) = (V'*train{i}(:,j))/norm(V'*train{i}(:,j));   
    end
end 

%% Classifier testing

for i = 1:L %project the dev and eval
    for j = 1:size(dev{i},2)
        w_dev{i}(:,j) = (V'*dev{i}(:,j))/norm(V'*dev{i}(:,j));   
    end
end

for i = 1:L  
    for j = 1:size(eval{i},2)
        w_eval{i}(:,j) = (V'*eval{i}(:,j))/norm(V'*eval{i}(:,j));   
    end
end   

L = 24;% redo part2
iv = w_train{1};
count_Ivector(1,1) = size(w_train{1},2);

for i = 2:L
    iv = [iv w_train{i}];
    count_Ivector(i,1) = size(w_train{i},2);
end  

for i = 1:L %calculate dev alpha
    for j = 1:size(w_dev{i},2)
        temp = lasso(iv,w_dev{i}(:,j),'NumLambda',3);
        alpha_deve3{(i-1)*100+j}(:,1) = temp(:,1);
    end
end

for j = 1:2400 %final cell
    count_alpha = 0;
    temp_alpha = alpha_deve3{j};
    for k = 1:L
        temp = zeros(size(iv,2),1);
        temp((count_alpha+1):(count_alpha+count_Ivector(k,1))) = temp_alpha((count_alpha+1):(count_alpha+count_Ivector(k,1)));
        count_alpha = count_alpha + count_Ivector(k,1);
        cell_deve3{j}(:,k) = temp(:,1);
    end
end

for i = 1:L %calculate eval alpha
    for j = 1:size(w_eval{i},2)
        temp = lasso(iv,w_eval{i}(:,j),'NumLambda',3);
        alpha_eval3{(i-1)*100+j}(:,1) = temp(:,1);
    end
end

for j = 1:2400 %final cell
    count_alpha = 0;
    temp_alpha = alpha_eval3{j};
    for k = 1:L
        temp = zeros(size(iv,2),1);
        temp((count_alpha+1):(count_alpha+count_Ivector(k,1))) = temp_alpha((count_alpha+1):(count_alpha+count_Ivector(k,1)));
        count_alpha = count_alpha + count_Ivector(k,1);
        cell_eval3{j}(:,k) = temp(:,1);
    end
end

%% Compute the residuals

for i = 1:L %dev residuals
    for j = 1:size(w_dev{i},2)
        dis_deve3{(i-1)*100+j} = w_dev{i}(:,j)-iv*cell_deve3{(i-1)*100+j};
        for k = 1:L
            res_deve3{(i-1)*100+j}(k,1) = norm(dis_deve3{(i-1)*100+j}(:,k));
        end
        
    end
end


for i = 1:L %eval residuals
    for j = 1:size(w_eval{i},2)
        dis_eval3{(i-1)*100+j} = w_eval{i}(:,j)-iv*cell_eval3{(i-1)*100+j};
        for k = 1:L
            res_eval3{(i-1)*100+j}(k,1) = norm(dis_eval3{(i-1)*100+j}(:,k));
        end
        
    end
end

%% identity and classify
count_deve3 = 0;
for i = 1:2400
    [~, idx1(i,1)] = min(res_deve3{i});
    if idivide(int32(i),int32(100),'ceil') == idx1(i,1)
        count_deve3 = count_deve3 + 1;
    end

end
acc_dev3 = count_deve3/2400; 

count_eval3 = 0;
for i = 1:2400
    [~, idx2(i,1)] = min(res_eval3{i});
    if idivide(int32(i),int32(100),'ceil') == idx2(i,1)
        count_eval3 = count_eval3 + 1;
    end

end
acc_eval3 = count_eval3/2400;

disp('The test accuracy for dev:');disp(acc_dev3);
disp('The test accuracy for eval:');disp(acc_eval3);
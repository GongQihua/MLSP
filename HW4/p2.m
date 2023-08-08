clc;
clear all;
close all;

%% Data loading 

train_path = "../Data/Train/Class_";
dev_path = "../Data/Dev/Class_";
eval_path = "../Data/Eval/Class_";

train = loaddata(train_path); %loading
dev = loaddata(dev_path);
eval = loaddata(eval_path);

%% Normalizing I-vector and calculate the corresponding alpha
%same like part1
L = 24;
iv = train{1};
count_Ivector(1,1) = size(train{1},2);
for i = 2:L
    iv = [iv train{i}];
    count_Ivector(i,1) = size(train{i},2);
end  

for i = 1:L %calculate dev alpha
    for j = 1:size(dev{i},2)
        temp = lasso(iv,dev{i}(:,j),'NumLambda',3);
        alpha_dev{(i-1)*100+j}(:,1) = temp(:,1);
    end
end

for j = 1:2400 %final cell
    count_alpha = 0;
    alpha = alpha_dev{j};
    for k = 1:L
        temp = zeros(size(iv,2),1);
        temp((count_alpha+1):(count_alpha+count_Ivector(k,1))) = alpha((count_alpha+1):(count_alpha+count_Ivector(k,1)));
        count_alpha = count_alpha + count_Ivector(k,1);
        cell_dev{j}(:,k) = temp(:,1);
    end
end

for i = 1:L %calculate eval alpha
    for j = 1:size(eval{i},2)
        temp = lasso(iv,eval{i}(:,j),'NumLambda',3);
        alpha_eval{(i-1)*100+j}(:,1) = temp(:,1);
    end
end

for j = 1:2400 %final cell
    count_alpha = 0;
    alpha = alpha_eval{j};
    for k = 1:L
        temp = zeros(size(iv,2),1);
        temp((count_alpha+1):(count_alpha+count_Ivector(k,1))) = alpha((count_alpha+1):(count_alpha+count_Ivector(k,1)));
        count_alpha = count_alpha + count_Ivector(k,1);
        cell_eval{j}(:,k) = temp(:,1);
    end
end

%% Compute the residuals

for i = 1:L %dev residuals
    for j = 1:size(dev{i},2)
        dis_dev{(i-1)*100+j} = dev{i}(:,j)-iv*cell_dev{(i-1)*100+j};
        for k = 1:L
            res_dev{(i-1)*100+j}(k,1) = norm(dis_dev{(i-1)*100+j}(:,k));
        end
        
    end
end

for i = 1:L %eval residuals
    for j = 1:size(eval{i},2)
        dis_eval{(i-1)*100+j} = eval{i}(:,j)-iv*cell_eval{(i-1)*100+j};
        for k = 1:L
            res_eval{(i-1)*100+j}(k,1) = norm(dis_eval{(i-1)*100+j}(:,k));
        end
        
    end
end

%% calculate accuracy
count_dev = 0;
for i = 1:2400
    [~, idx1(i,1)] = min(res_dev{i});
    if idivide(int32(i),int32(100),'ceil') == idx1(i,1)
        count_dev = count_dev + 1;
    end

end
acc_dev = count_dev/2400; 

count_eval = 0;
for i = 1:2400
    [~, idx2(i,1)] = min(res_eval{i});
    if idivide(int32(i),int32(100),'ceil') == idx2(i,1)
        count_eval = count_eval + 1;
    end
end
acc_eval = count_eval/2400;

disp('The test accuracy for dev:');disp(acc_dev);
disp('The test accuracy for eval:');disp(acc_eval);
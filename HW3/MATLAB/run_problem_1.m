%% Driver Script for HW2, Problem 2
% GMMs for Speaker Recognition
% Auther Name: Qihua Gong
clc; 
clear all; 
close all;
%% load data
training = cell(20,2); %training matrix
path = dir('../data/train/');
for files = 1 : length(path)
    if path(files).bytes < 1 % if its not real data, skip
        continue
    else
        training{files-2, 1} = importdata(['../data/train/',path(files).name]);
        training{files-2, 2} = path(files).name; %build the matrix with data and name
    end
end
%disp(data);
%% EM Algorithm for training GMMs
K = 64; 
num_s = 10; %speaker num
epochs = 20; 
finalmodel = cell(3,num_s); % Final model
likelihood = zeros(num_s,epochs);%likelihood
for s = 1 : num_s 
    X = [training{s*2-1}; training{s*2}];
    N = size(X,1);
    D = size(X,2);
    cluster = cell(4,K);

    for i = 1 : K %calculate clusters
        cluster{1,i} = 1/K;
        cluster{2,i} = rand(1,D);
        
        t = randn(20,20);
        cluster{3,i} = t'*t;
    end
    
    for e = 1 : epochs %training epochs
        denominator = zeros(N,1);
        for k = 1 : K %calculate denominator
            ak  = cluster{1,k};
            bk  = cluster{2,k};
            ck = cluster{3,k};
            [V,D2] = eig(ck); %calculate eig
            d = diag(D2);
            d(d<=1e-7) = 1e-7; % swap num under limit
            diagonal = diag(d);
            ck = (V*diagonal*V');  % calculate diagonal
            numerator = ak * mvnpdf(X, bk, ck); %build numerator and denominator
            denominator = denominator + numerator;
            cluster{4,k} = numerator; % save in cluster
        end

        for k = 1 : K  
            cluster{4,k} = cluster{4,k} ./ denominator; %divide numerator and denominator
            cluster{4,k}(isnan(cluster{4,k}))=0;
        end

        for k = 1 : K %gamma
            gamma_k = cluster{4,k};
            ck = zeros(D);
            nk = sum(gamma_k);
            ak = nk / N;  
            bk = sum(gamma_k .* X) / nk;

            for n = 1 : N
                diff = ( X(n,:)-bk )';
                ck = ck + (gamma_k(n) .* (diff * diff'));
            end
            ck = ck / nk;
           
            cluster{1,k} = ak;% save the new cluster
            cluster{2,k} = bk;
            cluster{3,k} = ck; 
        end
        Log = sum(log(denominator));%log likelihood
        likelihood(e,s) = Log; % track
        disp(['Training Epoch: ', num2str(e)]);
    end

    u_mat = zeros(K,D);%calculate u
    for k = 1 : K
        u_mat(k,:) = cluster{2,k}';
    end
    
    sigma = zeros(D,D,K);%sigma
    for k = 1 : K
        sigma(:,:,k) = cluster{3,k};
    end
    
    pi = zeros(K,1);%pi
    for k = 1 : K
        pi(k) = cluster{1,k};
    end
    test = sum(pi);
    disp(['Finish training for speaker ',num2str(s)]);
    finalmodel{1,s} = pi;% build final model matrix
    finalmodel{2,s} = u_mat;
    finalmodel{3,s} = sigma;     
end
save('model.mat','finalmodel')

%% Classification
% load test data
testing = cell(10,2);
path = dir('../data/test/');
for files = 1 : length(path)
    if path(files).bytes < 1
        continue
    else
        testing{files-2, 1} = importdata(['../data/test/',path(files).name]);
        testing{files-2, 2} = path(files).name;%build the matrix with data and name
    end
end
[testing{11,1}, testing{11,2}] = testing{2,:};%fix order
t = cell(10,2);
s1 = 0;
for i = 1 : length(testing)
    if i==2
        s1 = 1;
    elseif s1==1
        [t{i-1,1}, t{i-1,2}] = testing{i,:};
    else
        [t{i,1}, t{i,2}] = testing{i,:};
    end
end
testing = t;

%compute the log-likelihood for test
load('model')
K = 20;
Log2 = zeros(10,10);
for x = 1 : 10 
    for y = 1 : 10 
        X = testing{y};
        denominator = zeros(length(X),1);

        for k = 1 : K 
            ak  = finalmodel{1,x}(k,1);
            bk  = finalmodel{2,x}(k,:);
            ck = finalmodel{3,x}(:,:,k);
            [V,D2] = eig(ck); 
            d = diag(D2); %same process in the train
            d(d<=1e-7) = 1e-7;
            diagonal = diag(d);
            ck = (V*diagonal*V'); 
            numerator = ak * mvnpdf(X, bk, ck);
            denominator = denominator + numerator;
        end
        Log = sum(log(denominator)); %log likehood
        Log2(y, x) = Log;
    end
end

load('../data/utt2spk')%load utt and compare,use matlab function
tag= ["101188-m", "102147-m","103183-m", "106888-m","110667-m", "2042-f","3424-m", "4177-m","4287-f", "7722-f"];
[M2,I2]= max(Log2,[],2);
mLog = zeros(10,10);%calculate probabilities for the GMMs
pred = cell(10,3);%prediction
for i = 1 : 10 % test sets
    X_train = [training{i*2-1}; training{i*2}]; %use the given matlab function to calculate
    GMM = fitgmdist(X_train,64,'RegularizationValue',0.1);%fitgmdist
    for j = 1 : 10
        X_test = testing{j};
        [P,nlogL] = posterior(GMM, X_test);%posterior
        mLog(j,i) = -nlogL;
    end
    
    pred(i,1) = testing(i,2);
    pred(i,2) = {tag(I2(i))};
end
[~,I] = max(mLog,[],2);
for i = 1 : 10
    pred(i,3) = {tag(I(i))};
end
save('../results/test_predictions.txt','pred');%save
disp(pred);
count = 0;
for i = 1 : 10
    if pred{i,2}==pred{i,3}
        count = count + 1;
    end
end
accuracy = count/10;
disp("The final accuracy = " + accuracy);%compare
%compare the hand wirte function and the matlab build in, result: The final accuracy = 0.6
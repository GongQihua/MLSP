%% Lab 4 - Non-negative Matrix Factorization
% Author Name :Qihua Gong

% You can create as many auxilliary scripts as you want
% As long as we can run this script to get all the plots we require

%% Import Data
close all; clear all; clc;
d1 = 112; d2 = 92; d = d1*d2; 
num_people = 40; num_images = 9;
data_train = zeros(d, num_people*num_images);

count = 1;
for i = 1:num_people
    for j = 1:num_images
        filename = sprintf('./../orl_faces/Train/s%i/%i.pgm', i, j);
        img = double(imread(filename));
        
        data_train(:,count) = reshape(img, d, 1);
        count = count+1;
    end
end

V = data_train/max(data_train(:));


%% Performing NMF
[B, W, obj, k] = nmf(V, 40, 0.001, 500);


%% Validation on the ORL Faces Dataset
figure;
sgtitle('Basis functions obtained by NMF');
for k = 1:40
  subplot(5, 8, k);
  imagesc(reshape(B(:,k), d1, d2));
  colormap gray; 
  axis image off;
end

% Compare your results with MATLAB's predefined NMF function 
opt = statset('MaxIter', 500, 'Display', 'final'); 
[B, W] = nnmf(V, 40, 'options', opt, 'algorithm', 'mult');

figure();
sgtitle('Basis functions obtained by MATLAB NMF Function');
for k = 1:40
  subplot(5, 8, k);
  imagesc(reshape(B(:,k), d1, d2));
  colormap gray; 
  axis image off;
end


%% Performing NMF with Sparsity Constraints
[B, W, obj, k] = nmf_sparse(V, 40, 0.001, 500, 100, 1);


%% Validation on the ORL Faces Dataset
figure();
sgtitle('Basis functions obtained by Sparse NMF');
for k = 1:40
  subplot(5, 8, k);
  imagesc(reshape(B(:,k), d1, d2));
  colormap gray; axis image off;
end


%% Lab 5 - Image Segmentation through K-Means
% Author Name : Qihua Gong

% You can create as many auxilliary scripts as you want
% As long as we can run this script to get all the plots we require
close all; 
clear; 
clc;
%% Implementing K-Means
%[centroids, image_segmented] = KMeans( image, k, maxIter);


%% Validation on Test Images
elephant = imread('../data/elephant.jpg');
eiffel = imread('../data/eiffel.jpg');
maxIter = 100;

figure(1);
subplot(2,2,1);
imshow(elephant);
title('origin');

K = 2;
[~, segmented_image] = KMeans(elephant,K,maxIter);
subplot(2,2,2);
imshow(segmented_image);
title('k = 2');

K = 5;
[~, segmented_image] = KMeans(elephant,K,maxIter);
subplot(2,2,3);
imshow(segmented_image);
title('k = 5');

K = 10;
[~, segmented_image] = KMeans(elephant,K,maxIter);
subplot(2,2,4);
imshow(segmented_image);
title('k = 10');

figure(2);
subplot(2,2,1);
imshow(eiffel);
title('origin');

K = 2;
[~, segmented_image] = KMeans(eiffel,K,maxIter);
subplot(2,2,2);
imshow(segmented_image);
title('k = 2');

K = 5;
[~, segmented_image] = KMeans(eiffel,K,maxIter);
subplot(2,2,3);
imshow(segmented_image);
title('k = 5');

K = 10;
[~, segmented_image] = KMeans(eiffel,K,maxIter);
subplot(2,2,4);
imshow(segmented_image);
title('k = 10');

%% Comparison with Existing K-Means
figure(3);
subplot(1,3,1);
imshow(elephant);
title('origin');

K = 5;
L = imsegkmeans(elephant,K);
B = labeloverlay(elephant,L);
subplot(1,3,2);
imshow(B);
title('segmented from matlab');

[~, segmented_image] = KMeans(elephant,K,maxIter);
subplot(1,3,3);
imshow(segmented_image);
title('segmented from previous');
%% Code Optimization (Optional)
K = 5;
tic;
L = imsegkmeans(elephant,K);
B = labeloverlay(elephant,L);
disp('runing matlab build-in imsegkmenas');
toc;

tic;
[~, segmented_image] = KMeans(elephant,K,maxIter);
disp('runing lab KMeans');
toc;
%already optimize the code, the first version of the kmeans need to run 20
%more seconds
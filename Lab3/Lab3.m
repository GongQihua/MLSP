%% Lab 3: PCA-based Face Recognition
clc; 
clear all; 
close all;
%% Step 1: Load the training data
% Read all the training images, reshape them into a vector and store them into the columns of a 10304x360
% matrix where 360 is the total number of images and 10304 (=112*92) is the dimension of the vectors.

% Your code goes here
for i=1:1:40 
    path =strcat( './orl_faces/Train/s',num2str(i),'/');
    dirlist  = dir([path '*.pgm']); 
    for j = 1:length(dirlist)
        datapath = [path dirlist(j).name];
        data = imread(datapath);
        W1(:,:,(i*length(dirlist)-length(dirlist)+j)) = data;
    end
end
num = 360;
for i=1:1:num
    x = W1(:,:,i); 
    y = reshape(x,1,size(x,1)*size(x,2));
    W(:,i) = y;%reshape to whole train matrix
end

%% Step 2: Perform PCA to extract the eigen-faces
% 1. Center the data
% 2. Compute the correlation matrix
% 3. Use either the <SVD> or <eig> function to perform SVD/eigen-decomposition and get the eigenvectors 
% and eigenvalues of the correlation matrix.Please refer to your class lecture on PCM to figure out the 
% following things: 
%	- SVD/eigendecompositon of what?
%	- How to get the eigenfaces and the corresponding eigenvalues from SVD/eig.
% 4. Normalize the eigenvectors by their L2 norm if necessary

% Your code goes here
meanface = mean(W'); 
W = double(W);
for i=1:1:num
      W(:,i) = (W(:,i))-meanface'; %center the data
end
cor = (W)*(W');%correlation
[V,D] = eigs(cor,num);%use eig to eigenvectors and eigenvalues
for j=1:1:num
    V(:,j) = V(:,j)/norm(V(:,j),2); %normalize
end

%% Step 3: Plot the eigenvalues
% Sort the eigenvectors and eigenvalues in descending order (if they are not sorted). Then plot the eigenvalues.

% Your code goes here
for s=1:1:(40)
    d(s) = D(s,s); %reshape eigenvalues to 1 row
end
figure(1);
plot(1:1:40,d);
title('Eigenvalues');
xlabel('Order');
ylabel('Size');

%% Step 4: Plot the first 3 eigenfaces and the last eigenface
% Make sure that the eigenfaces are sorted in descending order of their eigenvalues.
% Remember that the images were mean-normalized and reshaped into vectors. So to plot the eigenfaces,
% you have to undo these steps.
% You can plot the images with- 
%   imagesc(image);
%   colormap gray;

% Your code goes here
eigenfaces = reshape(V ,size(x,1),size(x,2),num); % undo and reshape
figure(2);
subplot(2,2,1);
imagesc(eigenfaces(:,:,1));
colormap gray;
title('eigenface 1');
subplot(2,2,2);
imagesc(eigenfaces(:,:,2));
colormap gray;
title('eigenface 2');
subplot(2,2,3);
imagesc(eigenfaces(:,:,3));
colormap gray;
title('eigenface 3');
subplot(2,2,4);
imagesc(eigenfaces(:,:,360));
colormap gray;
title('eigenface 360');

%% Step 5: Pick a face and reconstruct it using k = {10, 20, 30, 40} eigenvectors. 
% Plot all of these reconstructions and compare them. For each value of k, plot the original image, 
% reconstructed image, and the difference between the original image and reconstruction in each case. 
% Write your observations in comments.

% Your code goes here
origin = double(W1(:,:,12));
M = double(origin);
T = reshape(M,1,size(x,1)*size(x,2));
T = T - meanface;
rec = zeros(1,size(x,1)*size(x,2));%reconstructed
for j =1:4
    rec = zeros(1,size(x,1)*size(x,2));
    k = j*10;%k = {10, 20, 30, 40}
    for i = 1:k
        W2(1,i) = V(:,i)'* T';
    end
    for i = 1:k
        rec = rec + W2(1,i) *  V(:,i)';
    end
    rec = reshape(rec + meanface,size(x,1),size(x,2));
    figure(j+2);
    subplot(1,3,1);
    imagesc(origin);
    colormap gray;
    title('Original Image');
    subplot(1,3,2);
    imagesc(rec);
    colormap gray;
    title({'Reconstruced Image'});
    subplot(1,3,3);
    imagesc(imbinarize((origin-rec)));
    title('Differenced Image');
end
% From my observation, the larger the k goes, the reconstructed graph
% become more clearly.

%% Step 6: Load the testing data, and reshape it similar to the training data.
% Subtract the training mean from the test images.
for j=1:1:40 
    path =strcat( './orl_faces/Test/s',num2str(j),'/');
    dirlist  = dir([path '*.pgm']); 
    for i = 1:length(dirlist)         
        data_test = imread([path dirlist(i).name]);
        W3(:,:,(j*length(dirlist)-length(dirlist)+i)) = data_test;
    end
end
for h=1:1:40
    x = W3(:,:,h); 
    y = reshape(x,1,size(x,1)*size(x,2));
    W_test(:,h) = y;%get the 10304*360 test matrix
end
%% Step 7: For each photograph in the testing dataset, predict the identity of the person.
% You will implement a classifier to do the following steps - 
% 1. Determine the projection of each test image onto k number of eigenfaces.
% 2. Compare the distance of each test image's projection to the projections of all images in the 
% training data.
% 3. Find the closest training image for each test image by finding the training projection with 
% minimum distance to the test projection.
% 4. Predict the identity of each test image by assigning it the identity of the closest training 
% image. Calculate and print the accuracy.
% Do the steps for each k = {10, 20, 30, 40}. Write the accuracies of all ks at the end in comments.

% Your code goes here
k = 40;
cor_P= cor(:,1:k);
test_P= cor_P*(inv(cor_P'*cor_P)*cor_P')*double(W_test);                                     
train_P = cor_P*(inv(cor_P'*cor_P)*cor_P')*double(W);
for i=1:40
    for j=1:360
        dis(j) = norm(test_P(:,i)-train_P(:,j) ,2);
    end
    proj_list(i)=find(dis==min((dis)));%build the projection compare list
end
cnt = 0;
for i=1:40
    if (proj_list(i)>=(i-1)*10) && (proj_list(i)<=i*10)
        cnt = cnt + 1;
    end
end
acc = cnt/40;
disp('accuracy = ');disp(acc);
%% Step 8: Show the closest image in the training dataset for the s1 test example.
% Plot the test image and the closest training images found using k = {10, 20, 30, 40}.

% Your code goes here
figure();
subplot(1,2,1);
imshow(mat2gray(W3(:,:,1)));
title('test image');
subplot(1,2,2);
imshow(mat2gray(W1(:,:,proj_list(1))));
title('Closest training image');
%saveas(gcf,'./results/closestimg_k10.png');
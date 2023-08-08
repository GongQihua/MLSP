%% Driver File for Problem 1: Simple Face Detector
% Implement a simple face detector that can detect faces in group photos 
% of people
% Author Name : Qihua Gong

%% Your Script Starts Here
% You can create as many auxilliary scripts as you want
% As long as we can run this script to get all the plots we require
clc;
clear all;
close all;
%% Compute the first eigenface from training data. And Scan the image.
path =strcat( '../../data/lfw_1000/');
filenames  = dir([path '*.pgm']); 
flat = 64*64; 
mat = [];
for i = 1 : length(filenames)%Read the data
    datapath = [path filenames(i).name];
    img = double(imread(datapath));
    mat(:,i) = reshape(img, flat, 1);
end
train = mat/max(mat(:));%get into a new matrix
center = train - mean(train, 'all');
corr = center * center';
[v, d] = eig(corr);% get eig
E = reshape(v(:,4096), [64 64]);
E2 = mat2gray(E);
[U,S,V] = svd(center,0);

%% Scan the image and Detect faces in an image.
path =strcat( '../../data/groups/');
I = imread([path 'Beatles.jpg']); 
I2 = squeeze(mean(I,3));
[P,Q] = size(I, [1 2]);
[N,M] = size(E, [1 2]);
for i = 1:(P-N) %scan range
    for j = 1:(Q-M)
        poj = I2(i:(i+N-1), j:(j+M-1));
        m(i,j) = E2(:)' * double(poj(:)) / norm(double(poj(:)));
    end
end
%% Scale the image and merge the peaks
K = [0.5, 0.75, 1.5, 2.0];% set different K
image = {};
count = 1;
num_iter = 40;
for f = K
    P2 = P*f; 
    Q2 = Q*f;
    scaled = imresize(I2,[P2,Q2]);
    image{2,count} = []; 
    image{3,count} = []; 
    image{4,count} = [];
    
    for i = 1:(P2-N) 
        for j = 1:(Q2-M) 
            poj = scaled(i:(i+N-1), j:(j+M-1));%get the patch size and score
            image{1,count}(i,j) = E2(:)' * double(poj(:)) / norm(double(poj(:)));
            
            if length(image{2,count}) < num_iter%merge peaks in different situation
                image{2,count} = [image{2,count} image{1,count}(i,j)];   
                image{3,count} = [image{3,count} reshape(poj, 4096, 1)];
                image{4,count} = [image{4,count}; [i j] ];
                
            elseif ismember(0, image{1,count}(i,j) >= image{2,count}) && length(image{2,count}) == num_iter
                [MIN, IDX] = min(image{2,count});
                image{2,count}(IDX) = image{1,count}(i,j);
                image{3,count}(:,IDX) = reshape(poj, 4096, 1);
                image{4,count}(IDX,:) = [i j];
            
            else
                continue
            
            end 
        end 
    end
    disp(f);
    count = count + 1;
end
%% Show the faces.
K = 2;
for face = 1 : length(image{2,1})%output
    subplot(8,5,face)
    hold on
    title(['face:' num2str(face)])
    imshow(reshape(image{3,K}(:,face), [64 64]), [])
    hold off
end

figure;
imshow(reshape(image{3,1}(:,35), [64 64]), [], ...
    'XData', [image{4,1}(35,1)*2 image{4,1}(35,1)*2+64],...
    'YData',[image{4,1}(35,2)*2 image{4,1}(35,2)*2+64])
axes('Position', [0, 0, 0, 0]);

imshow(I);
title('origin image')

figure;
subplot(1,4,1);
imshow(reshape(image{3,1}(:,35), [64 64]), []);
title('person 1');

subplot(1,4,2);
imshow(reshape(image{3,1}(:,23), [64 64]), []);
title('person 2');

subplot(1,4,3)
imshow(reshape(image{3,2}(:,40), [64 64]), []);
title('person 3');

subplot(1,4,4)
imshow(reshape(image{3,1}(:,28), [64 64]), []);
title('person 4');
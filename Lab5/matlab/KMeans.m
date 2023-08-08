function [centroid, image_segmented] = KMeans( image, k, max_iter)
% KMeans - Implement KMeans to segment an image
% 
% Arguments:
%   image    - Input array containing RGB colors of image
%   k        - Number of means, or centroids, to calculate
%   max_iter - Maximum number of iterations
% Returns:
%   centroid - RGB coordinates of the centroids
%   image_segmented - The input image segmented into k colors

% Your code here
%% load the data and Vectorize the image based on RGB components
Y = double(image);
[rows, cols, ~] = size(Y);
flat = rows * cols;
img = zeros(size(Y));
img(:,:,1) = Y(:, :, 1);
img(:,:,2) = Y(:, :, 2);
img(:,:,3) = Y(:, :, 3);
img = reshape(img, [flat 3] ); 
%% Initialize the centroids. Randomly select RGB points from the image as initial centroids.
randim = randperm(size(img,1)); 
centroid = img(randim(1:k), :);
%% Assign all data points to the closest centroid.
[m,n] = size(img);
for X = 1:max_iter %Repeat steps 4 and 5 until the stop criteria is reached
    cluster = zeros(m, 1); % initialize cluster
    for i=1:m %compute the distance
      K = 1;
      distance = sum((img(i,:) - centroid(1,:)) .^ 2);
      for j=2:k
          dist = sum((img(i,:) - centroid(j,:)) .^ 2);%check every point with the cluster
          if dist < distance %find the min distance
            distance = dist;
            K = j;
          end
      end
      cluster(i) = K;
    end
%% Re-compute the centroid vectors as the mean of all pixels assigned to each respective centroid.
    centroid = zeros(k,n);
    for i=1:k
      newdist = img(find(cluster==i),:);
      pix = size(newdist,1);
      centroid(i, :) = (1/pix) * sum(newdist);
    end
 
end
%% output and resegment
seg = reshape(cluster, [rows cols]); %reshape and out put the segmented
image_segmented = zeros(size(Y)); 
for i = 1:rows
    for j = 1:cols
        for d = 1:k
            if (seg(i,j) == d)
                image_segmented(i,j,:) = centroid(d,:);
            end
        end
    end
end
image_segmented = uint8(image_segmented);

end
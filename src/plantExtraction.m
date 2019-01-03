close all; 
clc;
clear all; 
im_file = '../crop-image/wheat01.jpg'; 
p = imread(im_file); 
p = imresize(p, 0.1);

p_hsv = rgb2hsv(p);

%split into 3 channel
p_red = double(p(:,:,1)); 
p_green = double(p(:,:,2)); 
p_blue = double(p(:,:,3)); 

%excess green image 
p_excess_green = 2 * p_green - p_red - p_blue;

%excess red image
p_excess_red = 1.4 * p_red - p_green; 

%NDI normalized difference index 
ndi = (p_green - p_red) / (p_green + p_red);

% Color index of vegetation (CIVE)
cive = 0.441 * p_red - 0.881 * p_green + 0.385* p_blue + 18.78745;

%vegetative 
vegetive = p_green / (p_red.^(.667) .* p_blue.^(.333));

%sobel 
sobel = edge(p_excess_green);

figure('name','original'), imshow(p);
figure('name','p_red'), imshow(uint8(p_red)); 
figure('name','p_excess_green'), imshow(uint8(p_excess_green));
figure('name','p_excess_red'), imshow(uint8(p_excess_red));
figure('name','NDI'), imagesc(ndi);
figure('name','cive'), imagesc(cive);
figure('name','vegetive'), imagesc(vegetive);
figure('name','hsv'), imshow(p_hsv);
figure('name','sobel'), imshow(sobel);

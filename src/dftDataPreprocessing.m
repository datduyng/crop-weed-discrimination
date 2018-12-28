close all; 
clc; 
clear all; 

data_dir = 'C:\Users\dnguyen52\Box\college\pitlaResearch\dataset\dataset\';


im1 = imread(sprintf('%sbroadleaf\\1.tif',data_dir));
im2 = imread(sprintf('%ssoybean\\29.tif',data_dir)); 


im1 = double(imresize(im1, [200 200]));
im2 = double(imresize(im2, [200 200]));

im1_dft = fft2(im1); 
im2_dft = fft2(im2);



figure('name','im1 original'), imshow(im1);
figure('name','im2 original'), imshow(im2);

figure('name','im1 original'), imshow(im1_dft);
figure('name','im2 original'), imshow(im2_dft);
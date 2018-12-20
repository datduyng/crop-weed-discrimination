close all; 
clear all;
o = imread('./crop-image/crop-weed01.png');
figure('name','Origin'), imshow(o); 
p = decorrstretch(o);
%Red Component of Colour Image
p_red=p(:,:,1);
%Green Component of Colour Image
p_green=p(:,:,2);
%Blue Component of Colour Image
p_blue=p(:,:,3);

p_crop = imcrop(p,[0 0 50 50]); 

%Apply Two Dimensional Discrete Wavelet Transform
[LLr,LHr,HLr,HHr]=dwt2(p_red,'haar');
[LLg,LHg,HLg,HHg]=dwt2(p_green,'haar');
[LLb,LHb,HLb,HHb]=dwt2(p_blue,'haar');

class(p_green)

% Compute excess green 
p_excess_green = 128 + (p_green - p_blue) + (p_green - p_red); 

mask = p_excess_green > 253; 

figure, imshow(mask);

% to decrease effect from light intensity, 
red_ratio = double(p_red) ./ double(p_red+p_blue+p_green); 

%masking and segmentate green out of red. 
%logic: if G > R and G > B -> output:1 else 0 
binary = (p_green > p_red); 
binary = (p_green > p_blue); 


% figure, imshow(red_ratio);
figure, imshow(p_excess_green); 
% figure, imshow(binary);
% figure, imshow(p_crop);
% figure('name','mask'), imshow(mask); 
% 
% 
% figure('name','Decorrelation stretch'), imshow(p); 

% 
% %%%%%%%%%red
% figure(1)
% subplot(2,2,1)
% imagesc(LLr)
% colormap gray
% title('Approximation')
% 
% subplot(2,2,2)
% imagesc(LHr)
% colormap gray
% title('Horizontal')
% 
% subplot(2,2,3)
% imagesc(HLr)
% colormap gray
% title('Vertical')
% 
% subplot(2,2,4)
% imagesc(HHr)
% title('Diagonal')
% colormap gray
% figure, imshow(p);

%% Function 
%{
base on: https://hadrienj.github.io/posts/Preprocessing-for-deep-learning/
1- Calculate the covariance matrix
2- Calculate the eigenvectors of the covariance matrix
3- Apply the matrix of eigenvectors to the data (this will apply the rotation)
%}
function decorrelated = decorrelate(p)
    [nrow ncol d] = size(p); 
    p = reshape(p,[nrow*ncol,d]); 
    
    cov = double(p') * double(p) ./ double(nrow);% size(p,2) -> dim 2 
    %cal the eigenvalue and eigen vector of covariance matrix 
    [eigVals,eigVecs] = eig(cov)
    %apply the eigen vector to input 
    
    decorrelated = double(p) * eigVecs;
    decorrelated =reshape(decorrelated,[nrow, ncol, d]);
end 
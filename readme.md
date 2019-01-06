## Plants and weeds discrimination model 
- **Project Status:**: On going..
- Projec

***Section***
- [Overview](#overview)
- [Working Flowchart](#working-flowchart)
- [Model break down](#model-break-down)


# Overview 
###### Goal: 
- Design a robust and simple to integrate plants and weeds machine learning model. 
- Proposes plants/weeds discriminant model that are able to process in real time. 

###### Why? 
- Machine learning vision is wide area of research. Automated Crops and weed control will save cost and reduce environment impact. 




## Working Flowchart


## Model break down

## Dataset 
- This image dataset has 15336 segments, being 3249 of soil, 7376 of 
soybean, 3520 grass and 1191 of broadleaf weeds.
- We want the model to uniformly learn all classes therefore,
	- training set: 4000 images(1000 each classes) 
	- validation set: 200 images(50 each classes)	
	- test: 200(50 each classes) 

## Difference Learning model on Dataset
- Determine plant coordinate in input images/video
	- [Plant blob coordinate](./experiment/Plants-weeds-blob-detection.ipynb)
- [Preprocessing and get Familiar with Dataset](https://www.kaggle.com/datduyn/preprocessing-plants-weeds-data-model/)
    - `10%` of Dataset is `testset `
- Problem: 
    - limited dataset. with roughly `10,000` more images each model below will achieve `7.5 - 15%` more
    
    
|			Model  		| Best Accuracy |Note|Status  |  
|-----------------------|---------------|----|--------|
|[Logistic Regression](https://www.kaggle.com/datduyn/logist-regression-on-plants-weeds-discrimination/edit)|59.5%|    |  |
|HSV space - 2 Layer net| 34.5%  				|Bad Overall    |
|[2 Layer Net](https://www.kaggle.com/datduyn/2-layer-net-on-weeds-discriminant/)|76.5%(Not stable)|    ||
|[Fully Connected Net](./experiment/Fully-Connected-Layer.ipynb)|78.5%(More stable)|||


## References 
- [Link to dataset](https://www.kaggle.com/fpeccia/weed-detection-in-soybean-crops)
###### Keyword
[Kaggle](kaggle.com) | Neural net | [Open CV](https://opencv.org/) | [Pytorch](https://pytorch.org/) | [Jupyter notebook](https://jupyter.org/)  | [Python3](https://www.python.org/) 


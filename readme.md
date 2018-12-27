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
- Determine plant coordinate in input images/video
	- [Plant blob coordinate](./experiment/Plants-weeds-blob-detection.ipynb)
- [Preprocessing and get Familiar with Dataset](https://www.kaggle.com/datduyn/preprocessing-plants-weeds-data-model/)
    - `10%` of Dataset is `testset `

|			Model  		| Best Accuracy |Note|Status  |  
|-----------------------|---------------|----|--------|
|[Logistic Regression](https://www.kaggle.com/datduyn/logist-regression-on-plants-weeds-discrimination/edit)|     57%		|    |Tuning  |
|[2 Layer Net](https://www.kaggle.com/datduyn/2-layer-net-on-weeds-discriminant/)|   			|    |On Going|
|   					|  				|    |

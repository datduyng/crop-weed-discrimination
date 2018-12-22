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
- [Plant blob coordinate](./experiment/Plants-weeds-blob-detection.ipynb)
```
Algorithm: 
Input: Image(RGB)
    1. Find contour -> array of vector
    2. Find min bounding box for each vector
    3. Remove overlap box 
    4. Tokenize each box from original image
```
- [Preprocessing and get Familiar with Dataset](./experiment/Classifier-Data-Preprocessing.ipynb)
    - `10%` of Dataset is testset 

- Machine Learning Technique on Image 
    - [Logistic Regression](Logistic-Regression-Plants-weeds-Model.ipynb) (On Going) 
        - Accuracy: 56.3 %(More improvement will be make on this) 


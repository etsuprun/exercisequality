---
title: "Predicting Excercise Quality from Accelerometer Data on Personal Fitness Devices (Course Project for Machine Learning)"

author: "Eugene Tsuprun"
date: "September 20, 2015"
output: html_document
---

# Overview

Hi, my name is Eugene Tsuprun. This is the course project on machine learning from JSU through Coursera.

We have a dataset with accelerometer data from six participants doing barbell lift exercises, along with a rating of whether they were doing the exercise correctly and, if not, the manner in which they were doing it incorrectly.  These ratings are included in the classe column of our dataset.

The objective is to build a model that predicts the rating of the manner the exercise was performed (i.e., the classe variable).


# Getting, Cleaning, and Preparing the Data

Many measurements in the dataset have mostly NA or blank values, so we'll go ahead and take those out of the model. We'll also take away the variables that we don't think will be valuable in out-of-sample prediction, such as timestamps and user IDs.

We'll separate the dataset into training and testing, with a ratio of 65% and 35% respectively.



```r
# Load the libraries and data
require(caret)
require(curl)
set.seed(100)


if(!file.exists("pml-training.csv")) {
  write.csv(pmldata<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),
    "pml-training.csv"
  )
} else {
  pmldata<-read.csv("pml-training.csv")  
}

# Take out columns that have NA values. It looks like all the columns that have any NA values have a lot of them, so we won't take them into account. 

pmldata<-pmldata[,!sapply(pmldata,function(x) any(is.na(x)))]

# Take out column where more than 90% of the values are blank.

pmldata<-pmldata[, !sapply(pmldata,function(x) (length(x[x==""])/length(x)>.9))]

# Take out this variable because only 406 out of 19622 values are "no" (the rest are "yes").
pmldata$new_window<-NULL

# Take out username, sequence number, and timestamps
pmldata$X<-NULL
pmldata$user_name<-NULL
pmldata$raw_timestamp_part_1<-NULL
pmldata$raw_timestamp_part_2<-NULL
pmldata$cvtd_timestamp<-NULL
pmldata$num_window<-NULL

# Partition data 

train.index<-createDataPartition(pmldata$classe, p=.65,list=F)
training<-pmldata[train.index, ]
testing<-pmldata[-train.index,]
```

# Building the Model

We train a random forest model on our 65% partition.


```r
# Load the random forest object if it already exists. 

if (file.exists("rf.fit.Rdata")) {
  load("rf.fit.Rdata")
} else {
  rf.fit<-train(classe~.,data=training,method="rf")
  save(rf.fit,file="rf.fit.Rdata")
}
```

# Evaluating the Model


```r
confusion<-confusionMatrix(predict(rf.fit,newdata=testing),testing$classe)
print(confusion)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1952    1    0    0    0
##          B    0 1326    0    0    0
##          C    1    1 1194    2    2
##          D    0    0    2 1123    2
##          E    0    0    1    0 1258
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9983          
##                  95% CI : (0.9969, 0.9991)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9978          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9995   0.9985   0.9975   0.9982   0.9968
## Specificity            0.9998   1.0000   0.9989   0.9993   0.9998
## Pos Pred Value         0.9995   1.0000   0.9950   0.9965   0.9992
## Neg Pred Value         0.9998   0.9996   0.9995   0.9997   0.9993
## Prevalence             0.2845   0.1934   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1932   0.1739   0.1636   0.1832
## Detection Prevalence   0.2845   0.1932   0.1748   0.1642   0.1834
## Balanced Accuracy      0.9996   0.9992   0.9982   0.9988   0.9983
```

The model is 99.83% accurate on the testing sample. The out-of-sample error rate is estimated at  0.17%.

Not bad.

Just for fun, let's compare it to a decision tree model and a gbm model.


```r
rpart.fit<-train(classe~.,data=training,method="rpart")

# Load the gbm model if it already exists. 

if (file.exists("gbm.fit.Rdata")) {
  load("gbm.fit.Rdata")
} else {
  gbm.fit<-train(classe~.,data=training,method="gbm")
  save(gbm.fit,file="gbm.fit.Rdata")
}

rpart.confusion<-confusionMatrix(predict(rpart.fit,newdata=testing),testing$classe)
gbm.confusion<-confusionMatrix(predict(gbm.fit,newdata=testing),testing$classe)
```

The decision tree model model is 66.16% accurate on the testing sample. The gbm model is 97.33% accurate.

Random forest wins at 99.83%.


# Reference

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3mwhi3Z8c

library(readr)
library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

train = read_csv('train.csv')

# create folds

set.seed(117)

fold1 <- train %>% slice_sample(n=150000)
remaining1  <- anti_join(train, fold1, by = 'id')

fold2 <- remaining1 %>% slice_sample(n=150000)
remaining2  <- anti_join(remaining1, fold2, by = 'id')

fold3 <- remaining2 %>% slice_sample(n=150000)
remaining3  <- anti_join(remaining2, fold3, by = 'id')

fold4 <- remaining3 %>% slice_sample(n=150000)
remaining4  <- anti_join(remaining3, fold4, by = 'id')

fold5 <- remaining4

rm(remaining, remaining1, remaining2, remaining3, remaining4)


### baseline polynomial model ###
# 0.4 ish

mod = glm(Calories ~ Age + I(Duration^2) + I(Heart_Rate^2) + I(Body_Temp^2), 
   rbind(fold1,fold2,fold3,fold4),
   family=poisson)

summary(mod)

yhat = predict(mod, fold5)

# Root Mean Squared Logarithmic Error

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

### other polynomial models ###

### through 5 degree poly of Duration ###

#randomly shuffle data
train.shuffled <- train[sample(nrow(train)),]

#define number of folds to use for k-fold cross-validation
K <- 10 

#define degree of polynomials to fit
degree <- 5

#create k equal-sized folds
folds <- cut(seq(1,nrow(train.shuffled)),breaks=K,labels=FALSE)

#create object to hold rmsle's of models
rmsle = matrix(data=NA,nrow=K,ncol=degree)

#Perform K-fold cross validation
for(i in 1:K){
  
  #define training and testing data
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- train.shuffled[testIndexes, ]
  trainData <- train.shuffled[-testIndexes, ]
  
  #use k-fold cv to evaluate models
  for (j in 1:degree){
    fit.train = glm(Calories ~ poly(Duration,j), data=trainData, family=poisson)
    fit.test = predict(fit.train, newdata=testData)
    rmsle[i,j] = sqrt(mean((log(1+fit.test)-log(1+testData$Calories))^2))
  }
}

#find MSE for each degree 
colMeans(rmsle)

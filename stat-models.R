library(readr)
library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)
library(MASS)

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
# gaussian, manually convert negatives to 0s
# 0.4 ish

mod = lm(Calories ~ Age + I(Duration^2) + I(Heart_Rate^2) + I(Body_Temp^2), 
   rbind(fold1,fold2,fold3,fold4))

summary(mod)

yhat = predict(mod, fold5)

# when < 0, predict 0

yhat = unname(yhat)

yhat[which(yhat<0)] <- 0

# Root Mean Squared Logarithmic Error

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))


### poisson model ###
# 2.6 ish

# check assumptions

# variance super inflated
mean(train$Calories)
var(train$Calories)

# within sex - not better
train %>% 
  filter(Sex == 'female') %>% 
  summarize(mean = mean(Calories),
            variance = var(Calories))

# model

mod = glm(Calories ~ Age + Duration + Heart_Rate + Body_Temp, 
          rbind(fold1,fold2,fold3,fold4),
          family="poisson")

summary(mod)

yhat = predict(mod, fold5)

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))


### through 5 degree poly of Duration ###
# gaussian
# 0.4 ish

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
    fit.train = lm(Calories ~ Age + poly(Duration,j) + I(Heart_Rate^2) + I(Body_Temp^2), data=trainData)
    fit.test = predict(fit.train, newdata=testData)
    # when < 0, predict 0
    fit.test = unname(fit.test)
    fit.test[which(fit.test<0)] <- 0
    # error calc
    rmsle[i,j] = sqrt(mean((log(1+fit.test)-log(1+testData$Calories))^2))
  }
}

#find MSE for each degree 
colMeans(rmsle)

### negative binomial ###
# 0.2 ish

summary(mod <- glm.nb(Calories ~ Age + Duration + Heart_Rate + Body_Temp, 
                     rbind(fold1,fold2,fold3,fold4),))

yhat = predict(mod, fold5, type = "response")

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

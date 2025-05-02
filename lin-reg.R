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

mod = lm(Calories ~ Age + Duration^2 + Heart_Rate^2 + Body_Temp^2, 
   rbind(fold1,fold2,fold3,fold4))

summary(mod)

yhat = predict(mod, fold5)

# when < 0, predict 0

yhat = unname(yhat)

yhat[which(yhat<0)] <- 0

# Root Mean Squared Logarithmic Error

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

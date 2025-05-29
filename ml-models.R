library(xgboost)
library(dplyr)
library(readr)


setwd(dirname(rstudioapi::getSourceEditorContext()$path))

train = read_csv('train.csv')

train <- train %>% 
  mutate(Female = case_when(Sex=='female' ~ 1,
                            Sex=='male'~0)) %>% 
  dplyr::select(!Sex)

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

rm(remaining1, remaining2, remaining3, remaining4)

### xgboost ###
# default booster is gbtree (tree-based)

train_x = rbind(fold1,fold2,fold3,fold4) %>% 
  dplyr::select(!c(id, Calories)) %>% 
  as.matrix()

train_y = rbind(fold1,fold2,fold3,fold4) %>% 
  dplyr::select(Calories) %>% 
  as.matrix()

test_x = fold5 %>% 
  dplyr::select(!c(id, Calories)) %>% 
  as.matrix()

# baseline
# 0.7 ish

mod <- xgboost(data = train_x, 
  label = train_y, 
  nrounds = 100, 
  objective = "reg:squaredlogerror")

yhat = predict(mod, test_x, validate_features = TRUE)

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

# 1000 rounds
# 0.6 ish (maxed out 1000 allowed rounds, default learning rate)

mod <- xgboost(data = train_x, 
               label = train_y, 
               nrounds = 1000, 
               objective = "reg:squaredlogerror",
               eval_set=0.2,
               #learning_rate=1,
               early_stopping_rounds = 1)

summary(mod)
mod$evaluation_log
mod$best_iteration

yhat = predict(mod, test_x, validate_features = TRUE)

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

# select best early_stopping_rounds (eval_set required) and learning_rate
# baseline for this will be actually getting it to early-stop
# 0.07 ish 
# (very wonky - only 10 evaluation obs and these overlap with val set, and only one early stop round)

train_mat = xgb.DMatrix(data=train_x,
                     label=train_y)

val_mat = xgb.DMatrix(data=as.matrix(fold5[1:10,c(2:7,9)]),
                      label=as.matrix(fold5[1:10,8]))

evals1 = list(train=train_mat, eval=val_mat)

mod <- xgb.train(data = train_mat,
                 nrounds = 1000,
                 watchlist = evals1,
                 early_stopping_rounds=1,
                 objective="reg:squaredlogerror")

yhat = predict(mod, test_x, validate_features = TRUE)

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

# select best early_stopping_rounds (eval_set required) and learning_rate
# test 5 learning rates at each value of early_stopping_rounds

# early stopping = 5

folds = list(fold1, fold2, fold3, fold4, fold5)

learning_rates = c(0.1, 0.3, 0.5, 0.7, 0.9)

es5_rmsles = c(99,99,99,99,99)

for (f in 1:5){
  
  print(f)
  
  # complete training data for this fold
  training_folds = train %>% 
    anti_join(folds[[f]], by=join_by(id))
  print('training folds complete')
  
  # eval data for this fold (subset of training)
  eval_index = sample(1:600000, 120000)
  
  eval_dataset = training_folds[eval_index,]
  
  eval_mat = xgb.DMatrix(data=as.matrix(eval_dataset[,c(2:7,9)]),
                        label=as.matrix(eval_dataset[,8]))
  
  print('eval data complete')
  
  # remaining training data
  train_dataset = training_folds[-eval_index,]
  
  train_mat = xgb.DMatrix(data=as.matrix(train_dataset[,c(2:7,9)]),
                          label=as.matrix(train_dataset[,8]))
  
  print('training data complete')
  

  # test data (f)
  test_x = folds[[f]] %>% 
    dplyr::select(!c(id, Calories)) %>% 
    as.matrix()
  
  print('test data complete')
  
  # model
  
  lr = learning_rates[f]
  print(lr)
  
  w = list(train=train_mat, eval=eval_mat)
  
  mod <- xgb.train(data = train_mat,
                   nrounds = 1000,
                   watchlist = w,
                   early_stopping_rounds=5,
                   objective="reg:squaredlogerror",
                   learning_rate=lr)
  
  print('model trained')
  
  yhat = predict(mod, test_x, validate_features = TRUE)
  
  print(sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2)))
  es5_rmsles[f]=sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2))
  
}

# early stopping = 10

es10_rmsles = c(99,99,99,99,99)

for (f in 1:5){
  
  print(f)
  
  # complete training data for this fold
  training_folds = train %>% 
    anti_join(folds[[f]], by=join_by(id))
  print('training folds complete')
  
  # eval data for this fold (subset of training)
  eval_index = sample(1:600000, 120000)
  
  eval_dataset = training_folds[eval_index,]
  
  eval_mat = xgb.DMatrix(data=as.matrix(eval_dataset[,c(2:7,9)]),
                        label=as.matrix(eval_dataset[,8]))
  
  print('eval data complete')
  
  # remaining training data
  train_dataset = training_folds[-eval_index,]
  
  train_mat = xgb.DMatrix(data=as.matrix(train_dataset[,c(2:7,9)]),
                          label=as.matrix(train_dataset[,8]))
  
  print('training data complete')
  

  # test data (f)
  test_x = folds[[f]] %>% 
    dplyr::select(!c(id, Calories)) %>% 
    as.matrix()
  
  print('test data complete')
  
  # model
  
  lr = learning_rates[f]
  print(lr)
  
  w = list(train=train_mat, eval=eval_mat)
  
  mod <- xgb.train(data = train_mat,
                   nrounds = 1000,
                   watchlist = w,
                   early_stopping_rounds=10,
                   objective="reg:squaredlogerror",
                   learning_rate=lr)
  
  print('model trained')
  
  yhat = predict(mod, test_x, validate_features = TRUE)
  
  print(sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2)))
  es10_rmsles[f]=sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2))
  
}

# early stopping = 15

es15_rmsles = c(99,99,99,99,99)

for (f in 1:5){
  
  print(f)
  
  # complete training data for this fold
  training_folds = train %>% 
    anti_join(folds[[f]], by=join_by(id))
  print('training folds complete')
  
  # eval data for this fold (subset of training)
  eval_index = sample(1:600000, 120000)
  
  eval_dataset = training_folds[eval_index,]
  
  eval_mat = xgb.DMatrix(data=as.matrix(eval_dataset[,c(2:7,9)]),
                         label=as.matrix(eval_dataset[,8]))
  
  print('eval data complete')
  
  # remaining training data
  train_dataset = training_folds[-eval_index,]
  
  train_mat = xgb.DMatrix(data=as.matrix(train_dataset[,c(2:7,9)]),
                          label=as.matrix(train_dataset[,8]))
  
  print('training data complete')
  
  
  # test data (f)
  test_x = folds[[f]] %>% 
    dplyr::select(!c(id, Calories)) %>% 
    as.matrix()
  
  print('test data complete')
  
  # model
  
  lr = learning_rates[f]
  print(lr)
  
  w = list(train=train_mat, eval=eval_mat)
  
  mod <- xgb.train(data = train_mat,
                   nrounds = 1000,
                   watchlist = w,
                   early_stopping_rounds=15,
                   objective="reg:squaredlogerror",
                   learning_rate=lr)
  
  print('model trained')
  
  yhat = predict(mod, test_x, validate_features = TRUE)
  
  print(sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2)))
  es15_rmsles[f]=sqrt(mean((log(1+yhat)-log(1+folds[[f]]$Calories))^2))
  
}

# use learning rate 0.3
# use 15 early stopping rounds


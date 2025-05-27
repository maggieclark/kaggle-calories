library(xgboost)

### xgboost baseline ###
# 0.7 ish

train_x = rbind(fold1,fold2,fold3,fold4) %>% 
  dplyr::select(!c(id, Calories)) %>% 
  as.matrix()

train_y = rbind(fold1,fold2,fold3,fold4) %>% 
  dplyr::select(Calories) %>% 
  as.matrix()

mod <- xgboost(data = train_x, 
  label = train_y, 
  nrounds = 100, 
  objective = "reg:squaredlogerror")

test_x = fold5 %>% 
  dplyr::select(!c(id, Calories)) %>% 
  as.matrix()

yhat = predict(mod, test_x, validate_features = TRUE)

sqrt(mean((log(1+yhat)-log(1+fold5$Calories))^2))

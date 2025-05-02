library(readr)
library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

train = read_csv('train.csv')
male = train %>% filter(Sex == 'male')
female = train %>% filter(Sex == 'female')

# even split male and female
# female category more calories
# male category more variance
# female category slightly higher dur and heart_rate which could explain some of the difference

table(train$Sex)[1]/nrow(train)

train %>% 
  group_by(Sex) %>% 
  summarize(median = median(Calories))

train %>% 
  ggplot(aes(x=Sex, y=Calories)) +
  geom_boxplot()

train %>% 
  group_by(Sex) %>% 
  summarize(median = median(Duration))

train %>% 
  group_by(Sex) %>% 
  summarize(median = median(Heart_Rate))

train %>% 
  group_by(Sex) %>% 
  summarize(median = median(Age))

# age is distributed linearly
# weak positive cor
# calorie variance increases with age (makes sense with linear age dist)

train %>%
  ggplot(aes(x=Age)) +
  geom_histogram()

train %>%
  sample_frac(0.001) %>% 
  ggplot(aes(x=Age, y=Calories)) +
  geom_point(position='jitter')

cor(train$Age, train$Calories)

# height normally distributed
# no cor

train %>%
  ggplot(aes(x=Height)) +
  geom_histogram()

train %>%
  sample_frac(0.05) %>% 
  ggplot(aes(x=Height, y=Calories)) +
  geom_point(position='jitter')

cor(train$Height, train$Calories)
cor(female$Height, female$Calories)
cor(male$Height, male$Calories)

# weight normally distributed
# no cor

train %>%
  ggplot(aes(x=Weight)) +
  geom_histogram()

train %>%
  sample_frac(0.05) %>% 
  ggplot(aes(x=Weight, y=Calories)) +
  geom_point(position='jitter')

cor(train$Weight, train$Calories)
cor(female$Weight, female$Calories)
cor(male$Weight, male$Calories)

# height*weight right skew distribution
  # normal when split on sex
# weight/height normal or bimodal
  # normal when split on sex
# no cor for either interation even when also splitting on sex

train %>%
  ggplot(aes(x=Weight*Height)) +
  geom_histogram() +
  facet_wrap(vars(Sex))

train %>%
  ggplot(aes(x=Weight/Height)) +
  geom_histogram()+
  facet_wrap(vars(Sex))

train %>%
  sample_frac(0.05) %>% 
  ggplot(aes(x=Weight*Height, y=Calories)) +
  geom_point(position='jitter')

train %>%
  sample_frac(0.05) %>% 
  ggplot(aes(x=Weight/Height, y=Calories)) +
  geom_point(position='jitter')

cor(train$Weight*train$Height, train$Calories)
cor(train$Weight/train$Height, train$Calories)
cor(female$Weight*female$Height, female$Calories)
cor(female$Weight/female$Height, female$Calories)
cor(male$Weight*male$Height, male$Calories)
cor(male$Weight/male$Height, male$Calories)

# duration uniform
# strong pos cor with increasing variance
# stronger cor btwn duration^2 and calories

train %>%
  ggplot(aes(x=Duration)) +
  geom_histogram()

train %>%
  sample_frac(0.05) %>% 
  ggplot(aes(x=Duration, y=Calories)) +
  geom_point(position='jitter')

cor(train$Duration, train$Calories)
cor(train$Duration^2, train$Calories)

# heart rate dist normal
# strong pos cor better modeled by exponential than linear

train %>%
  ggplot(aes(x=Heart_Rate)) +
  geom_histogram()

train %>%
  sample_frac(0.05) %>% 
  ggplot(aes(x=Heart_Rate, y=Calories)) +
  geom_point(position='jitter')

cor(train$Heart_Rate, train$Calories)
cor(train$Heart_Rate^2, train$Calories)

# body temp dist exponential
# strong pos cor better modeled by exponential than linear

train %>%
  ggplot(aes(x=Body_Temp)) +
  geom_histogram()

train %>%
  sample_frac(0.05) %>% 
  ggplot(aes(x=Body_Temp, y=Calories)) +
  geom_point(position='jitter')

cor(train$Body_Temp, train$Calories)
cor(train$Body_Temp^2, train$Calories)

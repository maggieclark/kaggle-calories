library(readr)
library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

train = read_csv('train.csv')
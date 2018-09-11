library(randomForest)
library(ggplot2)
library(reshape2)
library(Ecdat)
library(fastDummies)
library(readxl)
library(caret)

cc <- read_excel("C:\\Users\\clantroops\\Desktop\\creditcardclients.xls")

cc$ID <- NULL
colnames(cc)[24] <- "Target"

cols_factors <- c("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2","PAY_3", "PAY_4", "PAY_5", "PAY_6", "Target")
cc[cols_factors] <- lapply(cc[cols_factors], factor)

trainIndex <- createDataPartition(cc$Target, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train <- data.frame(cc[trainIndex,])

test <- data.frame(cc[-trainIndex,])

library(mlr)
traintask <- makeClassifTask(data = train,target = "Target")
testtask <- makeClassifTask(data = test,target = "Target")

rf.lrn <- makeLearner("classif.randomForest")
getParamSet(rf.lrn)

rf.lrn <- makeLearner(
  "classif.randomForest",
  predict.type = "response",
  par.vals = list(
    importance = TRUE
    
  )
)

#set parameter space
params <- makeParamSet(makeIntegerParam("mtry",lower = 1,upper = 10),
                       makeIntegerParam("ntree",lower = 500,upper = 550)
                  )
rdesc <- makeResampleDesc("CV",iters=5L)

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 200L)

#start tuning
tune <- tuneParams(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)

bst_rf <- randomForest(Target ~., data = train, importance = TRUE, proximity= TRUE, ntree =100, mtry = 4)
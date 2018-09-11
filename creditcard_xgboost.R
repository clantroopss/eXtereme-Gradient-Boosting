library(xgboost)
library(ggplot2)
library(reshape2)
library(Ecdat)
library(fastDummies)
library(readxl)
library(caret)

cc <- read_excel("C:\\Users\\clantroops\\Desktop\\creditcardclients.xls")

cc$ID <- NULL
colnames(cc)[24] <- "Target"

cols_factors <- c("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2","PAY_3", "PAY_4", "PAY_5", "PAY_6")
cc[cols_factors] <- lapply(cc[cols_factors], factor)

refined_dataset <- data.frame(dummy_columns(cc))

refined_dataset[cols_factors] <- NULL

refined_dataset$Target

refined_dataset$Target <- as.factor(refined_dataset$Target)
library(gmodels)
CrossTable(cc$SEX)

CrossTable(cc$PAY_2)

cc$MARRIAGE[cc$MARRIAGE == 0] <- NA

cc$EDUCATION[cc$EDUCATION == 0] <- NA

cc$PAY_6[cc$PAY_6 == 0] <- NA
cc$PAY_6[cc$PAY_6 == -2] <- NA


refined_dataset$Target1 <- refined_dataset$Target

refined_dataset$Target <- NULL
dim(refined_dataset)

trainIndex <- createDataPartition(refined_dataset$Target, p = .7, 
                                  list = FALSE, 
                                  times = 1)
train <- data.frame(refined_dataset[trainIndex,])

test <- data.frame(refined_dataset[-trainIndex,])

str(train)
dtrain <- xgb.DMatrix(data = as.matrix(train[,1:91]), label = as.matrix(train[,92]), missing = NA)

dtest <- xgb.DMatrix(data = as.matrix(test[,1:91]), label = as.matrix(test[,92]), missing = NA)

xgb_model <- xgb.cv( params= xgb_params_1, dtrain, nrounds = 500, nfold = 5)

xgb_params_1 = list(
  objective = "binary:logistic",                                               # binary classification
  eta = 0.3,                                                                  # learning rate
  max.depth = 6,                                                               # max tree depth
  eval_metric = "error"                                                          # evaluation/loss metric
)
?xgb.cv
library(mlr)

traintask <- makeClassifTask(data = train,target = "Target", positive = 1)
testtask <- makeClassifTask(data = test,target = "Target")

?makeClassifTask()

xgb.lrn <- makeLearner("classif.xgboost")

xgb.lrn <- makeLearner(
  "classif.xgboost",
  predict.type = "response",
  par.vals = list(
    objective = "binary:logistic",
    eval_metric = "error",
    early_stopping_rounds = 20,
    verbose = 0
  )
)

getParamSet(xgb.lrn)
rdesc <- makeResampleDesc("CV",iters=5L)



#set parameter space
params <- makeParamSet(makeIntegerParam("nrounds",lower = 500,upper = 1000),
                       makeIntegerParam("max_depth",lower = 1,upper = 10),
                       makeNumericParam("eta",lower = 0,upper = 1),
                       makeNumericParam("gamma",lower = 0,upper = 10),
                       makeNumericParam("subsample",lower = 0,upper = 1),
                       makeNumericParam("colsample_bytree",lower = 0,upper = 1),
                       makeNumericParam("lambda",lower = -1,upper = 0,trafo = function(x) 10^x)
)

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 200L)

#start tuning
tune <- tuneParams(learner = xgb.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)

tune$x
#[Tune] Result: nrounds=722; max_depth=2; eta=0.158; gamma=7.03; subsample=0.425; colsample_bytree=0.93; lambda=0.395 : acc.test.mean=0.8213895
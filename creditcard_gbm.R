library(gbm)
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

##################### GBM package ################################################

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = Target ~ .,
  distribution = "bernoulli",
  data = train,
  n.trees = 1000,
  interaction.depth = 1,
  shrinkage = 0.01,
  verbose = FALSE
  )  

# print results
print(gbm.fit)

################################# CARET GBM ################################################

caretGrid <- expand.grid(interaction.depth=c(1, 3, 5), n.trees = 500,
                         shrinkage=c(0.01, 0.001),
                         n.minobsinnode=10)
trainControl <- trainControl(method="cv", number=10)

set.seed(99)
gbm.caret <- train(Target ~ .,train, distribution="bernoulli", method="gbm",
                   trControl=trainControl, verbose=FALSE, 
                   tuneGrid=caretGrid,  bag.fraction=0.75)                  

print(gbm.caret)


######################### MLR GBM Implementation ##############################################
#### GRID Search ####




##### Random Search ######

library(mlr)
traintask <- makeClassifTask(data = train,target = "Target")
testtask <- makeClassifTask(data = test,target = "Target")


gbm.lrn <- makeLearner(
  "classif.gbm",
  predict.type = "response",
  par.vals = list(
    distribution = "bernoulli"
  )
)

rdesc <- makeResampleDesc("CV",iters=5L)

getParamSet(gbm.lrn)

#set parameter space
params <- makeParamSet(makeIntegerParam("n.trees",lower = 1000,upper = 5000),
                       makeIntegerParam("interaction.depth",lower = 1,upper = 3),
                       makeIntegerParam("n.minobsinnode",lower = 1,upper = 10),
                       makeNumericParam("shrinkage",lower = 0.001,upper = 0.3),
                       makeNumericParam("bag.fraction",lower = 0,upper = 1),
                       makeNumericParam("train.fraction",lower = 0,upper = 1)
                       )

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 200L)

#start tuning
tune <- tuneParams(learner = gbm.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)

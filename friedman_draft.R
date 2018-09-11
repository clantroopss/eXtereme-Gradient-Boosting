library(xgboost)
library(ggplot2)
library(reshape2)
library(Ecdat)
library(caret)
fried <- read.table("C:\\Users\\clantroops\\Desktop\\fried\\FriedmanExample\\fried_delve.data")
colnames(fried) <- c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","Target")

set.seed(1993)
trainIndex <- createDataPartition(fried$Target, p = .7, list = FALSE, times = 1)

dfTrain <- fried[trainIndex,]
dfTrain_matrix <- as.matrix(dfTrain)
train <- list(data = dfTrain_matrix[,1:10], label = dfTrain_matrix[,11])
dtrain <- xgb.DMatrix(data = train$data, label= train$label)

dfTest <- fried[-trainIndex,]
dfTest_matrix <- as.matrix(dfTest)
test <- list(data = dfTest_matrix[,1:10], label = dfTest_matrix[,11])
dtest <- xgb.DMatrix(data = test$data, label = test$label)


###############SINGLE MODEL################################################

watchlist <- list(train=dtrain, test=dtest)
bst <- xgb.train(data=dtrain, nrounds = 300, watchlist=watchlist, objective = "reg:linear",verbose = 2, max_depth =2)

xgb.dump(bst,with_stats = T)

xgb.plot.tree(colnames(train$data), bst)

xgb.ggplot.deepness(bst)
bst$params

lossfunc <- as.data.frame(bst$evaluation_log)
plot(c(0,100),c(min(lossfunc$test_rmse),max(lossfunc$test_rmse)),type='n', xlab = "Iteration", ylab = "RMSE")
lines(lossfunc$iter, lossfunc$train_rmse, col= "black", lwd=2.5)
lines(lossfunc$iter, lossfunc$test_rmse, col= "red", lwd=2.5)
legend((max(lossfunc$iter)-5),max(lossfunc$test_rmse), c('Train RMSE','Test RMSE'), lty=c(1,1), lwd=c(2.5,2.5),col=c('black','red'))

library(corrplot)

#by far best way to understand the model output using features it used
corrplot.mixed(cor(fried))

yhat_test <- predict(bst, test$data)
dfTest$yhat <- yhat_test
plot(test$label, yhat_test )
cat("Train  RMSE: " , sqrt(mean((yhat_test - test$label)^2)))

#plotting tree
xgb.plot.tree(feature_names = colnames(df), model = bst, trees = NULL)
xgb.plot.tree(feature_names = colnames(df), model = bst, trees = 0)

xgb.plot.tree(feature_names = colnames(df), model = bst, trees = 99)


#feature importance
importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

library(Ckmeans.1d.dp)
xgb.ggplot.importance(importance_matrix = importance_matrix)
xgb.ggplot.deepness(bst)

explainer <- xgboostExplainer::buildExplainer(bst,dtrain, type="regression", base_score =0.5)
xgboostExplainer::explainPredictions(bst,explainer = explainer,dtest)
xgboostExplainer::showWaterfall(bst,explainer = explainer,DMatrix=dtest,data.matrix = dfTest_matrix,type = "regression", idx = 5) ##Prediciton for 
#specific test data row
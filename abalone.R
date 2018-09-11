library(xgboost)
library(ggplot2)
library(reshape2)
library(Ecdat)
library(fastDummies)
abalone <- read.table("C:\\Users\\clantroops\\Desktop\\abalone.data", sep = ",")

abalone <- dummy_columns(abalone)

abalone$V1 <- NULL

abalone

abalone <- data.frame("V1_M" = abalone$V1_M, "V1_F" = abalone$V1_F, "V1_I" = abalone$V1_I, abalone)
# = select train and test indexes = #
df_train=abalone[1:3133,]
df_test= abalone[3134:4177,]

abalone[,12:14] <- NULL

input <- c("V1_M","V1_F","V1_I","V2","V3","V4","V5","V6", "V7", "V8")
target <- c("V9")

dfTrain_matrix <- as.matrix(df_train)
train <- list(data = dfTrain_matrix[,input], label = dfTrain_matrix[,target])


dtrain <- xgb.DMatrix(data = train$data, label= train$label)

dfTest_matrix <- as.matrix(df_test)
test <- list(data = dfTest_matrix[,1:8], label = dfTest_matrix[,9])

dtest <- xgb.DMatrix(data = test$data, label = test$label)

library(corrplot)
corrplot.mixed(cor(abalone))

###########################SIngle model ##################################################################################

watchlist <- list(train=dtrain, test=dtest)
system.time(bst <- xgb.train(data=dtrain, nrounds = 500, base_score = mean(df_train$V9),watchlist=watchlist, objective = "reg:linear",verbose = 2))
bst$params
lossfunc$train_rmse
lossfunc <- as.data.frame(bst$evaluation_log)
if(min(lossfunc$test_rmse) < min(lossfunc$train)) min = min(lossfunc$test_rmse) else min = min(lossfunc$train_rmse)
if (max(lossfunc$test_rmse) > max(lossfunc$train)) max = max(lossfunc$test_rmse) else max = max(lossfunc$train_rmse)

plot(c(0,100),c(min,max),type='n', xlab = "Iteration", ylab = "RMSE")
lines(lossfunc$iter, lossfunc$train_rmse, col= "black", lwd=2.5)
lines(lossfunc$iter, lossfunc$test_rmse, col= "red", lwd=2.5)
legend((max(lossfunc$iter)-5),max(lossfunc$test_rmse), c('Train RMSE','Test RMSE'), lty=c(1,1), lwd=c(2.5,2.5),col=c('black','red'))

########################### XGB CV Method ###############################################################################

bst_cv <- xgb.cv(data=dtrain, nrounds = 10000, base_score = mean(df_train$V9), objective = "reg:linear",verbose = 2, nfold = 5, eta = 0.001)
bst_cv$params
lossfunc$train_rmse_mean
lossfunc <- as.data.frame(bst_cv$evaluation_log)
if(min(lossfunc$test_rmse_mean) < min(lossfunc$train_rmse_mean)) min = min(lossfunc$test_rmse_mean) else min = min(lossfunc$train_rmse_mean)
if (max(lossfunc$test_rmse_mean) > max(lossfunc$train_rmse_mean)) max = max(lossfunc$test_rmse_mean) else max = max(lossfunc$train_rmse_mean)

plot(c(0,10000),c(min,max),type='n', xlab = "Iteration", ylab = "RMSE Cross Validation")
lines(lossfunc$iter, lossfunc$train_rmse_mean, col= "black", lwd=2.5)
lines(lossfunc$iter, lossfunc$test_rmse_mean, col= "red", lwd=2.5)
legend((max(lossfunc$iter)-20),max(lossfunc$test_rmse_mean), c('Train RMSE','Test RMSE'), lty=c(1,1), lwd=c(2.5,2.5),col=c('black','red'))

################################ Plain GBM implementation ##################################################################
library(gbm, quietly=TRUE)

# Get the time to train the GBM model
system.time(
  gbm.model <- gbm(V9 ~ .
                   , distribution = "gaussian"
                   , data = rbind(df_train, df_test)
                   , n.trees = 500
                   , interaction.depth = 1
                   , n.minobsinnode = 1
                   , shrinkage = 0.3
                   , bag.fraction = 1
                   ,cv= 5
  )
)

library(ROCR)
library(pROC)
?gbm.perf
# Determine best iteration based on test data
best.iter = gbm.perf(gbm.model, method = "test")

gbm.model$trees

# Get feature importance
gbm.feature.imp = summary(gbm.model, n.trees = best.iter)

# Plot and calculate AUC on test data
gbm.test = predict(gbm.model, newdata = df_test, n.trees = best.iter)
auc.gbm = roc(df_test$V9, gbm.test, plot = TRUE, col = "red")
print(auc.gbm)
############################################################################################################################


# = parameters = #
nrounds = c(100, 200, 300, 400, 500)
# = eta candidates = #
eta=c(0.01,0.05,0.1,0.2,0.3)
# = colsample_bylevel candidates = #
cs=c(1/3,2/3,1)
# = max_depth candidates = #
md=c(100,200,3000,40000)
# = sub_sample candidates = #
ss=c(0.25,0.5,0.75,1)
# = min_child_weights candidates = #
mcw=c(1,10,100,400)
# = gamma candidates = #
gamma=c(0,2,10)


# = standard model is the second value  of each vector above = #
standard=c(2,2,3,2,4)
#############################
xgb_grid_1 = expand.grid(
  eta = c(0.01, 0.03 ,0.05 ,0.1,0.3,1),                #[2-10]/num trees
  max_depth = 6,             #Start with 6
  nrounds = 500,                      #Fix at 100
  gamma = 0,                          #Usually ok to leave at 0
  colsample_bytree = 1,   #.3 - .5
  min_child_weight = 1,
  subsample =1 #start with 1/sqrt(eventrate)
)
# Tuning control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                         # save losses across all models
  allowParallel = TRUE
)


# Train the model on each set of parameters in the grid and evaluate using cross-validation
xgb_train_1 = caret::train(
  x = train$data,
  y = train$label,
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree",
  metric = "RMSE"
)
?plot.train()
plot(xgb_train_1)
xgb_train_1$results
#######################################################
best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0

for (iter in 1:10) {
  param <- list(objective = "reg:linear",
                eval_metric = "rmse",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.5), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1)
  )
  cv.nround = 100
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain, params = param, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stopping_rounds=8, maximize=FALSE)
  
  min_rmse = min(mdcv$evaluation_log[, test_rmse_mean])
  min_rmse_index = which.min(mdcv$evaluation_log[, test_rmse_mean])
  
  if (min_rmse < best_rmse) {
    best_rmse = min_rmse
    best_rmse_index = min_rmse_index
    best_seednumber = seed.number
    best_param = param
  }
}
mdcv$evaluation_log

nround = best_rmse_index
set.seed(best_seednumber)
md <- xgb.train(data=dtrain, params=best_param, nrounds=nround)
#############################
set.seed(1993)
conv_eta_train <- NULL
conv_eta_test <- NULL
pred_eta <- NULL
pred_eta_xgboost <- NULL

conv_eta_train = matrix(NA,500,length(eta))
conv_eta_test = matrix(NA,500,length(eta))
pred_eta = matrix(NA,nrow(df_train), length(eta))
pred_eta_xgboost = matrix(NA,nrow(df_train), length(eta))
colnames(conv_eta_train)= colnames(conv_eta_test) = colnames(pred_eta) = colnames(pred_eta_xgboost) = eta
for(i in 1:length(eta)){
  params=list(eta = eta[i], colsample_bylevel=1,
              subsample = 1, max_depth = 6,
              min_child_weigth = 1)
  
  xgb=xgb.cv(data = dtrain, nrounds = 500, params = params, nfold = 10,prediction = T)
  conv_eta_train[,i] = xgb$evaluation_log$train_rmse_mean
  conv_eta_test[,i] = xgb$evaluation_log$test_rmse_mean
  #pred_eta[,i] = predict(xgb, dtest)
  pred_eta_xgboost[,i] = xgb$pred
}
(RMSE_eta = sqrt(colMeans((train$label-data.frame(xgb$pred))^2)))

head(conv_eta_train)
head(conv_eta_train_1)
conv_eta_train_1 = data.frame(iter=1:500, conv_eta_train)
conv_eta_train_1 = melt(conv_eta_train_1, id.vars = "iter")
colnames(conv_eta_train_1)[2] <- "ETA"
ggplot(data = conv_eta_train_1) + geom_line(aes(x = iter, y = value, color = ETA)) + ggtitle("ETA") + xlab("Boosting Iteration") + 
  ylab("Train RMSE")

conv_eta_test_1 = data.frame(iter=1:500, conv_eta_test)
conv_eta_test_1 = melt(conv_eta_test_1, id.vars = "iter")
colnames(conv_eta_test_1)[2] <- "ETA"
ggplot(data = conv_eta_test_1) + geom_line(aes(x = iter, y = value, color = ETA)) + ggtitle("ETA") + xlab("Boosting Iteration") + 
  ylab("RMSE(Cross - Validation)")

wwhich(conv_eta_test == min(conv_eta_test, na.rm = TRUE), arr.ind = TRUE)
apply(conv_eta_test,2,which.min)

bar <- subset(conv_eta_train_1, ETA == "X0.3")
bar1 <- subset(conv_eta_test_1, ETA == "X0.3")

plot(c(0,10),c(0,10),type='n', xlab = "Iteration", ylab = "RMSE")
lines(bar$iter, bar$value, col= "black", lwd=2.5)
lines(bar1$iter, bar1$value, col= "red", lwd=2.5)

conv_eta_test = data.frame(iter=1:100, conv_eta_test)
conv_eta_test = melt(conv_eta_test, id.vars = "iter")
ggplot(data = conv_eta_test) + geom_line(aes(x = iter, y = value, color = variable))+ ggtitle("ETA") + xlab("Boosting Iteration") + 
  ylab("Test RMSE")


(RMSE_eta = sqrt(colMeans((df_train$V9-pred_eta_xgboost)^2)))

#looking at colsample_bylevel

set.seed(1993)
conv_cs_train <- NULL
conv_cs_test <- NULL
pred_cs <- NULL
conv_cs_train = matrix(NA,100,length(cs))
conv_cs_test = matrix(NA,100,length(cs))
pred_cs = matrix(NA,nrow(df_test), length(cs))
colnames(conv_cs_train)= colnames(conv_cs_test) = colnames(pred_cs) = cs
for(i in 1:length(cs)){
  params = list(eta = 0.3, colsample_bylevel = cs[i], max_depth = md[standard[3]],
                min_child_weigth = 1)
  xgb=xgb.train(data = dtrain, nrounds = 100, params = params,watchlist=watchlist)
  conv_cs_train[,i] = xgb$evaluation_log$train_rmse
  conv_cs_test[,i] = xgb$evaluation_log$test_rmse
  pred_cs[,i] = predict(xgb, dtest)
}
conv_cs_train = data.frame(iter=1:100, conv_cs_train)
conv_cs_train = melt(conv_cs_train, id.vars = "iter")
ggplot(data = conv_cs_train) + geom_line(aes(x = iter, y = value, color = variable))+ ggtitle("Coulmn Subsampling by level") + xlab("Boosting Iteration") + 
  ylab("Test RMSE")

conv_cs_test = data.frame(iter=1:100, conv_cs_test)

conv_cs_test = melt(conv_cs_test, id.vars = "iter")
ggplot(data = conv_cs_test) + geom_line(aes(x = iter, y = value, color = variable))+ ggtitle("ETA") + xlab("Boosting Iteration") + 
  ylab("Test RMSE")

(RMSE_cs = sqrt(colMeans((df_test$V9-pred_cs)^2)))

#looking at colsample_bylevel

set.seed(1993)
conv_cs_train <- NULL
conv_cs_test <- NULL
pred_cs <- NULL
conv_cs_train = matrix(NA,100,length(cs))
conv_cs_test = matrix(NA,100,length(cs))
pred_cs = matrix(NA,nrow(df_test), length(cs))
colnames(conv_cs_train)= colnames(conv_cs_test) = colnames(pred_cs) = cs
for(i in 1:length(cs)){
  params = list(eta = 0.3, colsample_bytree = cs[i], max_depth = md[standard[3]],
                min_child_weigth = 1)
  xgb=xgb.train(data = dtrain, nrounds = 100, params = params,watchlist=watchlist)
  conv_cs_train[,i] = xgb$evaluation_log$train_rmse
  conv_cs_test[,i] = xgb$evaluation_log$test_rmse
  pred_cs[,i] = predict(xgb, dtest)
}
conv_cs_train = data.frame(iter=1:100, conv_cs_train)
conv_cs_train = melt(conv_cs_train, id.vars = "iter")
ggplot(data = conv_cs_train) + geom_line(aes(x = iter, y = value, color = variable))+ ggtitle("Coulmn Subsampling by level") + xlab("Boosting Iteration") + 
  ylab("Test RMSE")

conv_cs_test = data.frame(iter=1:100, conv_cs_test)

conv_cs_test = melt(conv_cs_test, id.vars = "iter")
ggplot(data = conv_cs_test) + geom_line(aes(x = iter, y = value, color = variable))+ ggtitle("ETA") + xlab("Boosting Iteration") + 
  ylab("Test RMSE")

(RMSE_cs = sqrt(colMeans((df_test$V9-pred_cs)^2)))



# lloking at mx_depth

set.seed(1993)
conv_md_train <- NULL
conv_md_test <- NULL
pred_md <- NULL
conv_md_train =data.frame(matrix(NA,100,2*length(gamma)))
conv_md_test =matrix(NA,100,length(gamma))

pred_md=matrix(NA,nrow(df_test),length(gamma))
colnames(conv_md_train) =c("0.1_train","0.1_test","1_train", "1_test","10_train", "10_test")
a = 1
for(i in 1:length(gamma)){
  
  params=list(eta=0.3,colsample_bytree=1,
              subsample=1,max_depth=6,lambda = 1, gamma= gamma[i],
              min_child_weigth=1)
  xgb=xgb.cv(data = dtrain, nfold= 5,predict=T,nrounds = 100, params = params)
  
  conv_md_train[,names(conv_md_train)[a]] = xgb$evaluation_log$train_rmse_mean
  a= a+1
  conv_md_train[,names(conv_md_train)[a]] = xgb$evaluation_log$test_rmse_mean
  a= a+1
}

conv_md_train = data.frame(iter=1:100, conv_md_train)
conv_md_train = melt(conv_md_train, id.vars = "iter")
plot(c(0,100),c(0,8),type='n', xlab = "Iteration", ylab = "RMSE")
lines(conv_md_train$iter, head(conv_md_train$X0.1_train), col= "black", lwd=2.5)
lines(conv_md_train$iter, conv_md_train$X0.1_test, col= "red", lwd=2.5)

plot(c(0,100),c(0,8),type='n', xlab = "Iteration", ylab = "RMSE")
lines(conv_md_train$iter, conv_md_train$X1_train, col= "black", lwd=2.5)
lines(conv_md_train$iter, conv_md_train$X1_test, col= "red", lwd=2.5)

plot(c(0,100),c(0,8),type='n', xlab = "Iteration", ylab = "RMSE")
lines(conv_md_train$iter, conv_md_train$X100_train, col= "black", lwd=2.5)
lines(conv_md_train$iter, conv_md_train$X100_test, col= "red", lwd=2.5)


legend((max(lossfunc$iter)-5),max(lossfunc$test_rmse), c('Train RMSE','Test RMSE'), lty=c(1,1), lwd=c(2.5,2.5),col=c('black','red'))

conv_md_train = data.frame(iter=1:100, conv_md_train)
conv_md_train = melt(conv_md_train, id.vars = "iter")
ggplot(data = conv_md_train) + geom_line(aes(x = iter, y = value))+ ggtitle("lambda") + xlab("Boosting Iteration") + 
  ylab("Train RMSE")



conv_md_test = data.frame(iter=1:100, conv_md_test)
conv_md_test = melt(conv_md_test, id.vars = "iter")
ggplot(data = conv_md_test) + geom_line(aes(x = iter, y = value, color = variable))+ ggtitle("Max Depth") + xlab("Boosting Iteration") + 
  ylab("Test RMSE")


(RMSE_md = sqrt(colMeans((df_test$V9-pred_md)^2)))



#sub_sample

set.seed(1993)
conv_ss_train =matrix(NA,100,length(ss))
conv_ss_test = matrix(NA,100,length(ss))
pred_ss=matrix(NA,nrow(dtest),length(ss))
colnames(conv_ss_train)= colnames(conv_ss_test)=colnames(pred_ss)=ss
for(i in 1:length(ss)){
  params=list(eta=0.3,colsample_bytree=1,
              subsample=ss[i], max_depth=md[standard[3]],
              min_child_weigth=1)
  xgb=xgb.train(data = dtrain, nrounds = 100, params = params,watchlist=watchlist)
  conv_ss_train[,i] = xgb$evaluation_log$train_rmse
  conv_ss_test[,i] = xgb$evaluation_log$test_rmse
  pred_ss[,i] = predict(xgb, dtest)
}

conv_ss_train=data.frame(iter=1:100,conv_ss_train)
conv_ss_train=melt(conv_ss_train,id.vars = "iter")
ggplot(data=conv_ss_train)+geom_line(aes(x=iter,y=value,color=variable))+ ggtitle("Subsampling") + xlab("Boosting Iteration") + 
  ylab("Train RMSE")


conv_ss_test = data.frame(iter=1:100, conv_ss_test)
conv_ss_test = melt(conv_ss_test, id.vars = "iter")
ggplot(data = conv_ss_test) + geom_line(aes(x = iter, y = value, color = variable))+ ggtitle("Subsampling") + xlab("Boosting Iteration") + 
  ylab("Train RMSE")


(RMSE_md = sqrt(colMeans((df_test$V9-pred_ss)^2)))


#  min_child_weight

set.seed(1)
conv_mcw = matrix(NA,500,length(mcw))
pred_mcw = matrix(NA,length(test), length(mcw))
colnames(conv_mcw) = colnames(pred_mcw) = mcw
for(i in 1:length(mcw)){
  params = list(eta = 0.1, colsample_bylevel=2/3,
                subsample = 1, max_depth = 6,
                min_child_weight = mcw[i], gamma = 0)
  xgb = xgb.cv(xtrain, label = ytrain, nrounds = 500, params = params)
  conv_mcw[,i] = xgb$evaluation_log$train_rmse
  pred_mcw[,i] = predict(xgb, xtest)
}

#setting gamma

set.seed(1)
conv_gamma = matrix(NA,500,length(gamma))
pred_gamma = matrix(NA,nrow(dtest), length(gamma))
colnames(conv_gamma) = colnames(pred_gamma) = gamma
for(i in 1:length(gamma)){
  params = list(eta = 0.1, colsample_bylevel=2/3,
                subsample = 1, max_depth = 6, min_child_weight = 1,
                gamma = gamma[i])
  xgb = xgboost(dtrain, nrounds = 500, params = params)
  xgb.ggplot.deepness(xgb)
  conv_gamma[,i] = xgb$evaluation_log$train_rmse
  pred_gamma[,i] = predict(xgb, dtest)
}

conv_gamma = data.frame(iter=1:500, conv_gamma)
conv_gamma = melt(conv_gamma, id.vars = "iter")
ggplot(data = conv_gamma) + geom_line(aes(x = iter, y = value, color = variable))+ ggtitle("Gamma") + xlab("Boosting Iteration") + 
  ylab("Train RMSE")


(RMSE_md = sqrt(colMeans((df_test$V9-pred_gamma)^2)))

#################### Evalluate GBM and XGBoost for Abalone Dataset ###############################################
require(xgboost)
require(gbm)
require(methods)

?gbm
input
df_train[input]
 gbm.time = system.time({
   gbm.model <- gbm(V9 ~ V1_M+V1_F+V1_I+V2+V3+V4+V5+V6+V7+V8, data = df_train, n.trees = 500, 
                    interaction.depth = 6, shrinkage = 0.1, bag.fraction = 1,
                    verbose = TRUE)
 })

 print(gbm.time)
 
xgboost.time = list()
threads = c(1,2,4,8)
for (i in 1:length(threads)){
  thread = threads[i]
  xgboost.time[[i]] = system.time({
    xgmat <- xgb.DMatrix(data = train$data, label= train$label)
    param <- list("objective" = "reg:linear",
                  "bst:eta" = 0.1,
                  "bst:max_depth" = 6,
                  "eval_metric" = "rmse",
                  "silent" = 1,
                  "nthread" = thread)
    watchlist <- list("train" = dtrain)
    nround = 500
    print ("loading data end, start to boost trees")
    bst = xgb.train(param, xgmat, nround);
    # save out model
    xgb.save(bst, "abalone.model")
    print ('finish training')
  })
}

xgboost.time



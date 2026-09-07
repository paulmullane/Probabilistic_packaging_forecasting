library(readxl)
library(dplyr)
library(bsts)
library(Metrics)
library(e1071)
library(randomForest)
library(glmnet)
library(forecast)
library(xgboost)
library(glmnet)
library(Cubist)
library(arm)

#functions----
lasso_covariate_selection <- function(modelling_data_set){
  #this funciton uses LASSO regression to see what variables are relevant
  #variables with non zero coefficients are retained for modelling
  X <- as.matrix(modelling_data_set[ , c('Population', 'GDP', 'co2', 
                                         'material_footprint', 'exports', 
                                         'energy_consumption')])
  Y <- modelling_data_set$gpw
  set.seed(123)
  lasso_cv <- cv.glmnet(X, Y, alpha=1, nfolds=5) #cross validation to find optimal lambda
  lasso_fit <- glmnet(X, Y, alpha=1, lambda=lasso_cv$lambda.1se)
  
  return(coef(lasso_fit))
}

rolling_cv_svm <- function(train_data, covariates){
  #this function returns the optimal svm hyperparameters using a rolling window
  #cv approach. It looks to minimise rolling validation MAPE
  initial_window <- 15
  n <- nrow(train_data)
  
  cost_values <- seq(0.1, 5, by=0.1)
  gamma_values <- seq(0, 0.5, 0.05)
  epsilon_values <- seq(0.001, 0.1, by=0.01)
  
  results <- expand.grid(cost=cost_values, gamma=gamma_values, epsilon=epsilon_values)
  results$cv_mape <- NA
  
  formula <- as.formula(paste("gpw ~", paste(covariates, collapse=" + ")))
  
  for(i in 1:nrow(results)){
    cost <- results$cost[i]
    gamma <- results$gamma[i]
    epsilon <- results$epsilon[i]
    
    fold_mapes <- c()
    
    for(end in initial_window:(n-1)){
      
      train_fold <- train_data[1:end, ]
      val_fold   <- train_data[end+1, ]
      
      set.seed(123)
      svm_model <- svm(formula, data=train_fold, kernel="radial",
                       cost=cost, gamma=gamma, epsilon=epsilon)
      
      pred <- predict(svm_model, val_fold)
      fold_mapes <- c(fold_mapes, mape(val_fold$gpw, pred))
    }
    
    results$cv_mape[i] <- mean(fold_mapes)
  }
  
  best <- results[which.min(results$cv_mape), ]
  return(best)
}

rolling_cv_cubist <- function(train_data, covariates){
  #this function returns the optimal cubist hyperparameters using a rolling
  #window cv approach. It looks to minimise rolling validation MAPE
  
  initial_window <- 15
  n <- nrow(train_data)
  
  committees_values <- c(1, 5, 10, 20, 50)
  neighbors_values <- c(0, 1, 3, 5, 7, 9)
  
  results <- expand.grid(committees=committees_values, neighbors=neighbors_values)
  results$cv_mape <- NA
  
  X_train_full <- as.matrix(train_data[ , covariates])
  y_train_full <- train_data$gpw
  
  set.seed(123)
  for(i in 1:nrow(results)){
    committees <- results$committees[i]
    neighbors <- results$neighbors[i]
    
    fold_mapes <- c()
    
    for(end in initial_window:(n-1)){
      
      X_fold <- X_train_full[1:end, , drop=FALSE]
      y_fold <- y_train_full[1:end]
      X_val  <- X_train_full[end+1, , drop=FALSE]
      y_val  <- y_train_full[end+1]
      
      cubist_model <- cubist(x=X_fold, y=y_fold, committees=committees)
      
      pred <- predict(cubist_model, newdata=as.data.frame(X_val), neighbors=neighbors)
      fold_mapes <- c(fold_mapes, mape(y_val, pred))
    }
    
    results$cv_mape[i] <- mean(fold_mapes)
  }
  
  best <- results[which.min(results$cv_mape), ]
  return(best)
}

#read waste data-----
Glass_Packaging_Waste <- read_excel("Desktop/Packaging waste forecasting/Packaging Data/glass_packaging_waste.xlsx")
Glass_Packaging_Waste$TIME <- as.double(Glass_Packaging_Waste$TIME)

#read country data----
#goes through and reads in each country, filters on correct years, and adds the 
#gpw column to the country datasets
countries <- c("Austria", "Belgium", "Denmark", "Finland", "France", "Germany",
               "Ireland", "Italy", "Luxembourg", "Netherlands", "Portugal", "Spain", "Sweden")
vars <- c("Year", "Population", "GDP", "material_footprint", "energy_consumption", "co2", "exports")

modelling_data <- lapply(setNames(countries, tolower(countries)), function(ctry) {
  df <- read_excel(paste0("Desktop/Packaging waste forecasting/Country data/", ctry, ".xlsx")) |>
    dplyr::select(all_of(vars)) |>
    filter(Year >= 1997, Year <= 2022)
  df$gpw <- Glass_Packaging_Waste[[ctry]][Glass_Packaging_Waste$TIME <= 2022]
  df
})
list2env(modelling_data, envir = .GlobalEnv)


#austria----
#train/test split (80/20)
austria_train <- austria[1:21, ]
austria_test <- austria[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(austria_train)


#xarima
austria_xarima <- auto.arima(y=austria_train$gpw, max.p=5, max.q=5,
                             xreg=as.matrix(austria_train[, c('exports')]))
mape(austria_xarima$fitted, austria_train$gpw)
rmse(austria_xarima$fitted, austria_train$gpw)
mae(austria_xarima$fitted, austria_train$gpw)

austria_xarima_forecast <- forecast(austria_xarima, h=nrow(austria_test),
                                    xreg=as.matrix(austria_test[, c('exports')]))

mape(austria_xarima_forecast$mean, austria_test$gpw)
rmse(austria_xarima_forecast$mean, austria_test$gpw)
mae(austria_xarima_forecast$mean, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main= 'austria gpw recycling rate - XARIMA', col='black')
lines(y=austria_xarima$fitted, x=austria_train$Year, col='red', type='b')
lines(y=austria_xarima_forecast$mean, x=austria_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~exports, 
                           data=austria_train)
austria_bsts <- bsts(gpw~exports,
                     state.specification=ss, data=austria_train, niter=1000)

austria_bsts_fitted <- predict(austria_bsts, newdata=austria_train,
                               horizon=nrow(austria_train), burn=100)
mape(austria_bsts_fitted$mean, austria_train$gpw)
rmse(austria_bsts_fitted$mean, austria_train$gpw)
mae(austria_bsts_fitted$mean, austria_train$gpw)

austria_bsts_forecast <- predict(austria_bsts, newdata=austria_test, burn=100)
mape(austria_bsts_forecast$mean, austria_test$gpw)
rmse(austria_bsts_forecast$mean, austria_test$gpw)
mae(austria_bsts_forecast$mean, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main='austria gpw Recycling - BSTS', col='black')
lines(y=austria_bsts_fitted$mean, x=austria_train$Year, col='red', type='b')
lines(y=austria_bsts_forecast$mean, x=austria_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(austria_train, c('exports'))
proc.time() - start

austria_svm <- svm(gpw~exports, data=austria_train, kernel='radial',
                   cost=4, gamma=0.05, epsilon=0.001)
mape(austria_svm$fitted, austria_train$gpw)
rmse(austria_svm$fitted, austria_train$gpw)
mae(austria_svm$fitted, austria_train$gpw)

austria_svm_forecast <- predict(austria_svm, newdata=austria_test)
mape(austria_svm_forecast, austria_test$gpw)
rmse(austria_svm_forecast, austria_test$gpw)
mae(austria_svm_forecast, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main='austria gpw Recycling - SVM', col='black')
lines(y=austria_svm$fitted, x=austria_train$Year, col='red', type='b')
lines(y=austria_svm_forecast, x=austria_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(austria_train, c('exports'))
proc.time()-start

set.seed(123)
austria_rf <- randomForest(gpw~exports, data=austria_train, ntree=150,
                           maxnodes=8)
mape(austria_rf$predicted, austria_train$gpw)
rmse(austria_rf$predicted, austria_train$gpw)
mae(austria_rf$predicted, austria_train$gpw)

austria_rf_forecast <- predict(austria_rf, newdata=austria_test)
mape(austria_rf_forecast, austria_test$gpw)
rmse(austria_rf_forecast, austria_test$gpw)
mae(austria_rf_forecast, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main='austria gpw Recycling - Radom Forest')
lines(y=austria_rf$predicted, x=austria_train$Year, type='b', col='red')
lines(y=austria_rf_forecast, x=austria_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(austria_train, c('exports'))
proc.time()-start


austria_X_train <- as.matrix(austria_train[ , c('exports')])
austria_y_train <- austria_train$gpw
austria_X_test  <- as.matrix(austria_test[ , c('exports')])
austria_y_test  <- austria_test$gpw
austria_dtrain <- xgb.DMatrix(data=austria_X_train, label=austria_y_train)
austria_dtest  <- xgb.DMatrix(data=austria_X_test)

set.seed(123)
austria_xgb <- xgboost(data=austria_dtrain, nrounds=100, max_depth=2,
                       eta=0.1, lambda=0.1, verbose=0)

austria_xgb_fitted <- predict(austria_xgb, austria_dtrain)
mape(austria_xgb_fitted, austria_train$gpw)
rmse(austria_xgb_fitted, austria_train$gpw)
mae(austria_xgb_fitted, austria_train$gpw)

austria_xgb_forecast <- predict(austria_xgb, austria_dtest)
mape(austria_xgb_forecast, austria_test$gpw)
rmse(austria_xgb_forecast, austria_test$gpw)
mae(austria_xgb_forecast, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='gpw', main='austria gpw - XGBoost')
lines(y=austria_xgb_fitted, x=austria_train$Year, type='b', col='red')
lines(y=austria_xgb_forecast, x=austria_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#nnet
start <- proc.time()
rolling_cv_nnet(austria_train, c('exports'))
proc.time()-start

set.seed(123)
austria_nnet <- nnetar(austria_train$gpw, p=1, size=2, decay=0.1, 
                       repeats=10, xreg=as.matrix(austria_train[, c('exports')]))

mape(as.numeric(fitted(austria_nnet)[2:21]), austria_train$gpw[2:21])
rmse(as.numeric(fitted(austria_nnet)[2:21]), austria_train$gpw[2:21])
mae(as.numeric(fitted(austria_nnet)[2:21]), austria_train$gpw[2:21])

austria_nnet_forecast <- as.numeric(forecast(austria_nnet, h=3, 
                                             xreg=as.matrix(austria_test[, c('exports')]))$mean)

mape(austria_nnet_forecast, austria_test$gpw)
rmse(austria_nnet_forecast, austria_test$gpw)
mae(austria_nnet_forecast, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main='Austria gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(austria_nnet)), x=austria_train$Year, type='b', col='red')
lines(y=austria_nnet_forecast, x=austria_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(austria_train, c('exports'))
proc.time() - start

austria_X_train <- as.matrix(austria_train[, c('exports')])
austria_X_test <- as.matrix(austria_test[, c('exports')])

austria_enet <- glmnet(austria_X_train, austria_train$gpw, alpha=0.6, lambda=2)

austria_enet_fitted <- as.numeric(predict(austria_enet, newx=austria_X_train, s=2))
mape(austria_enet_fitted, austria_train$gpw)
rmse(austria_enet_fitted, austria_train$gpw)
mae(austria_enet_fitted, austria_train$gpw)

austria_enet_forecast <- as.numeric(predict(austria_enet, newx=austria_X_test, s=2))
mape(austria_enet_forecast, austria_test$gpw)
rmse(austria_enet_forecast, austria_test$gpw)
mae(austria_enet_forecast, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main='Austria gpw Recycling - Elastic Net', col='black')
lines(y=austria_enet_fitted, x=austria_train$Year, col='red', type='b')
lines(y=austria_enet_forecast, x=austria_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(austria_train, c('exports'))
proc.time() - start

austria_cubist <- cubist(x=as.matrix(austria_train[, c('exports')]),
                         y=austria_train$gpw, committees=5)

austria_cubist_fitted <- predict(austria_cubist, 
                                 newdata=as.data.frame(austria_train[, c('exports')]),
                                 neighbors=1)
mape(austria_cubist_fitted, austria_train$gpw)
rmse(austria_cubist_fitted, austria_train$gpw)
mae(austria_cubist_fitted, austria_train$gpw)

austria_cubist_forecast <- predict(austria_cubist,
                                   newdata=as.data.frame(austria_test[, c('exports')]),
                                   neighbors=1)
mape(austria_cubist_forecast, austria_test$gpw)
rmse(austria_cubist_forecast, austria_test$gpw)
mae(austria_cubist_forecast, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main='Austria gpw Recycling - Cubist', col='black')
lines(y=austria_cubist_fitted, x=austria_train$Year, col='red', type='b')
lines(y=austria_cubist_forecast, x=austria_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
austria_bayes_ridge <- bayesglm(gpw~exports, data=austria_train,
                                family=gaussian)

austria_bayes_fitted <- predict(austria_bayes_ridge, newdata=austria_train)
mape(austria_bayes_fitted, austria_train$gpw)
rmse(austria_bayes_fitted, austria_train$gpw)
mae(austria_bayes_fitted, austria_train$gpw)

austria_bayes_forecast <- predict(austria_bayes_ridge, newdata=austria_test)
mape(austria_bayes_forecast, austria_test$gpw)
rmse(austria_bayes_forecast, austria_test$gpw)
mae(austria_bayes_forecast, austria_test$gpw)

plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main='Austria gpw Recycling - Bayesian Ridge', col='black')
lines(y=austria_bayes_fitted, x=austria_train$Year, col='red', type='b')
lines(y=austria_bayes_forecast, x=austria_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#belgium----
#train/test split (80/20)
belgium_train <- belgium[1:21, ]
belgium_test <- belgium[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(belgium_train)


#xarima
belgium_xarima <- auto.arima(y=belgium_train$gpw, max.p=5, max.q=5,
                             xreg=as.matrix(belgium_train[, c('GDP')]))
mape(belgium_xarima$fitted, belgium_train$gpw)
rmse(belgium_xarima$fitted, belgium_train$gpw)
mae(belgium_xarima$fitted, belgium_train$gpw)

belgium_xarima_forecast <- forecast(belgium_xarima, h=nrow(belgium_test),
                                    xreg=as.matrix(belgium_test[, c('GDP')]))

mape(belgium_xarima_forecast$mean, belgium_test$gpw)
rmse(belgium_xarima_forecast$mean, belgium_test$gpw)
mae(belgium_xarima_forecast$mean, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main= 'belgium gpw recycling rate - XARIMA', col='black')
lines(y=belgium_xarima$fitted, x=belgium_train$Year, col='red', type='b')
lines(y=belgium_xarima_forecast$mean, x=belgium_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~GDP, 
                           data=belgium_train)
belgium_bsts <- bsts(gpw~GDP,
                     state.specification=ss, data=belgium_train, niter=1000)

belgium_bsts_fitted <- predict(belgium_bsts, newdata=belgium_train,
                               horizon=nrow(belgium_train), burn=100)
mape(belgium_bsts_fitted$mean, belgium_train$gpw)
rmse(belgium_bsts_fitted$mean, belgium_train$gpw)
mae(belgium_bsts_fitted$mean, belgium_train$gpw)

belgium_bsts_forecast <- predict(belgium_bsts, newdata=belgium_test, burn=100)
mape(belgium_bsts_forecast$mean, belgium_test$gpw)
rmse(belgium_bsts_forecast$mean, belgium_test$gpw)
mae(belgium_bsts_forecast$mean, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main='belgium gpw Recycling - BSTS', col='black')
lines(y=belgium_bsts_fitted$mean, x=belgium_train$Year, col='red', type='b')
lines(y=belgium_bsts_forecast$mean, x=belgium_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(belgium_train, c('GDP'))
proc.time() - start

belgium_svm <- svm(gpw~GDP, data=belgium_train, kernel='radial',
                   cost=5, gamma=0.5, epsilon=0.001)
mape(belgium_svm$fitted, belgium_train$gpw)
rmse(belgium_svm$fitted, belgium_train$gpw)
mae(belgium_svm$fitted, belgium_train$gpw)

belgium_svm_forecast <- predict(belgium_svm, newdata=belgium_test)
mape(belgium_svm_forecast, belgium_test$gpw)
rmse(belgium_svm_forecast, belgium_test$gpw)
mae(belgium_svm_forecast, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main='belgium gpw Recycling - SVM', col='black')
lines(y=belgium_svm$fitted, x=belgium_train$Year, col='red', type='b')
lines(y=belgium_svm_forecast, x=belgium_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(belgium_train, c('GDP'))
proc.time()-start

set.seed(123)
belgium_rf <- randomForest(gpw~GDP, data=belgium_train, maxnodes=6, ntree=100)

mape(belgium_rf$predicted, belgium_train$gpw)
rmse(belgium_rf$predicted, belgium_train$gpw)
mae(belgium_rf$predicted, belgium_train$gpw)

belgium_rf_forecast <- predict(belgium_rf, newdata=belgium_test)
mape(belgium_rf_forecast, belgium_test$gpw)
rmse(belgium_rf_forecast, belgium_test$gpw)
mae(belgium_rf_forecast, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main='belgium gpw Recycling - Radom Forest')
lines(y=belgium_rf$predicted, x=belgium_train$Year, type='b', col='red')
lines(y=belgium_rf_forecast, x=belgium_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(belgium_train, c('GDP'))
proc.time()-start


belgium_X_train <- as.matrix(belgium_train[ , c('GDP')])
belgium_y_train <- belgium_train$gpw
belgium_X_test  <- as.matrix(belgium_test[ , c('GDP')])
belgium_y_test  <- belgium_test$gpw
belgium_dtrain <- xgb.DMatrix(data=belgium_X_train, label=belgium_y_train)
belgium_dtest  <- xgb.DMatrix(data=belgium_X_test)

set.seed(123)
belgium_xgb <- xgboost(data=belgium_dtrain, nrounds=100, max_depth=3,
                       eta=0.3, lambda=0.5, verbose=0)

belgium_xgb_fitted <- predict(belgium_xgb, belgium_dtrain)
mape(belgium_xgb_fitted, belgium_train$gpw)
rmse(belgium_xgb_fitted, belgium_train$gpw)
mae(belgium_xgb_fitted, belgium_train$gpw)

belgium_xgb_forecast <- predict(belgium_xgb, belgium_dtest)
mape(belgium_xgb_forecast, belgium_test$gpw)
rmse(belgium_xgb_forecast, belgium_test$gpw)
mae(belgium_xgb_forecast, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='gpw', main='belgium gpw - XGBoost')
lines(y=belgium_xgb_fitted, x=belgium_train$Year, type='b', col='red')
lines(y=belgium_xgb_forecast, x=belgium_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(belgium_train, c('GDP'))
proc.time()-start

set.seed(123)
belgium_nnet <- nnetar(belgium_train$gpw, p=1, size=3, decay=0, 
                       repeats=10, xreg=as.matrix(belgium_train[, c('GDP')]))

mape(as.numeric(fitted(belgium_nnet)[2:21]), belgium_train$gpw[2:21])
rmse(as.numeric(fitted(belgium_nnet)[2:21]), belgium_train$gpw[2:21])
mae(as.numeric(fitted(belgium_nnet)[2:21]), belgium_train$gpw[2:21])

belgium_nnet_forecast <- as.numeric(forecast(belgium_nnet, h=3, 
                                             xreg=as.matrix(belgium_test[, c('GDP')]))$mean)

mape(belgium_nnet_forecast, belgium_test$gpw)
rmse(belgium_nnet_forecast, belgium_test$gpw)
mae(belgium_nnet_forecast, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main='belgium gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(belgium_nnet)), x=belgium_train$Year, type='b', col='red')
lines(y=belgium_nnet_forecast, x=belgium_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#elastic net
start <- proc.time()
rolling_cv_enet(belgium_train, c('GDP'))
proc.time() - start

belgium_X_train <- as.matrix(belgium_train[, c('GDP')])
belgium_X_test <- as.matrix(belgium_test[, c('GDP')])

belgium_enet <- glmnet(belgium_X_train, belgium_train$gpw, alpha=0.6, lambda=2)

belgium_enet_fitted <- as.numeric(predict(belgium_enet, newx=belgium_X_train, s=2))
mape(belgium_enet_fitted, belgium_train$gpw)
rmse(belgium_enet_fitted, belgium_train$gpw)
mae(belgium_enet_fitted, belgium_train$gpw)

belgium_enet_forecast <- as.numeric(predict(belgium_enet, newx=belgium_X_test, s=2))
mape(belgium_enet_forecast, belgium_test$gpw)
rmse(belgium_enet_forecast, belgium_test$gpw)
mae(belgium_enet_forecast, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main='belgium gpw Recycling - Elastic Net', col='black')
lines(y=belgium_enet_fitted, x=belgium_train$Year, col='red', type='b')
lines(y=belgium_enet_forecast, x=belgium_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(belgium_train, c('GDP'))
proc.time() - start

belgium_cubist <- cubist(x=as.matrix(belgium_train[, c('GDP')]),
                         y=belgium_train$gpw, committees=50)

belgium_cubist_fitted <- predict(belgium_cubist, 
                                 newdata=as.data.frame(belgium_train[, c('GDP')]),
                                 neighbors=1)
mape(belgium_cubist_fitted, belgium_train$gpw)
rmse(belgium_cubist_fitted, belgium_train$gpw)
mae(belgium_cubist_fitted, belgium_train$gpw)

belgium_cubist_forecast <- predict(belgium_cubist,
                                   newdata=as.data.frame(belgium_test[, c('GDP')]),
                                   neighbors=1)
mape(belgium_cubist_forecast, belgium_test$gpw)
rmse(belgium_cubist_forecast, belgium_test$gpw)
mae(belgium_cubist_forecast, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main='belgium gpw Recycling - Cubist', col='black')
lines(y=belgium_cubist_fitted, x=belgium_train$Year, col='red', type='b')
lines(y=belgium_cubist_forecast, x=belgium_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
belgium_bayes_ridge <- bayesglm(gpw~GDP, data=belgium_train,
                                family=gaussian)

belgium_bayes_fitted <- predict(belgium_bayes_ridge, newdata=belgium_train)
mape(belgium_bayes_fitted, belgium_train$gpw)
rmse(belgium_bayes_fitted, belgium_train$gpw)
mae(belgium_bayes_fitted, belgium_train$gpw)

belgium_bayes_forecast <- predict(belgium_bayes_ridge, newdata=belgium_test)
mape(belgium_bayes_forecast, belgium_test$gpw)
rmse(belgium_bayes_forecast, belgium_test$gpw)
mae(belgium_bayes_forecast, belgium_test$gpw)

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main='belgium gpw Recycling - Bayesian Ridge', col='black')
lines(y=belgium_bayes_fitted, x=belgium_train$Year, col='red', type='b')
lines(y=belgium_bayes_forecast, x=belgium_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#denmark----
#train/test split (80/20)
denmark_train <- denmark[1:21, ]
denmark_test <- denmark[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(denmark_train)


#xarima
denmark_xarima <- auto.arima(y=denmark_train$gpw, max.p=5, max.q=5,
                             xreg=as.matrix(denmark_train[, c('Population')]))
mape(denmark_xarima$fitted, denmark_train$gpw)
rmse(denmark_xarima$fitted, denmark_train$gpw)
mae(denmark_xarima$fitted, denmark_train$gpw)

denmark_xarima_forecast <- forecast(denmark_xarima, h=nrow(denmark_test),
                                    xreg=as.matrix(denmark_test[, c('Population')]))

mape(denmark_xarima_forecast$mean, denmark_test$gpw)
rmse(denmark_xarima_forecast$mean, denmark_test$gpw)
mae(denmark_xarima_forecast$mean, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main= 'denmark gpw recycling rate - XARIMA', col='black')
lines(y=denmark_xarima$fitted, x=denmark_train$Year, col='red', type='b')
lines(y=denmark_xarima_forecast$mean, x=denmark_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~Population, 
                           data=denmark_train)
denmark_bsts <- bsts(gpw~Population,
                     state.specification=ss, data=denmark_train, niter=1000)

denmark_bsts_fitted <- predict(denmark_bsts, newdata=denmark_train,
                               horizon=nrow(denmark_train), burn=100)
mape(denmark_bsts_fitted$mean, denmark_train$gpw)
rmse(denmark_bsts_fitted$mean, denmark_train$gpw)
mae(denmark_bsts_fitted$mean, denmark_train$gpw)

denmark_bsts_forecast <- predict(denmark_bsts, newdata=denmark_test, burn=100)
mape(denmark_bsts_forecast$mean, denmark_test$gpw)
rmse(denmark_bsts_forecast$mean, denmark_test$gpw)
mae(denmark_bsts_forecast$mean, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main='denmark gpw Recycling - BSTS', col='black')
lines(y=denmark_bsts_fitted$mean, x=denmark_train$Year, col='red', type='b')
lines(y=denmark_bsts_forecast$mean, x=denmark_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(denmark_train, c('Population'))
proc.time() - start

denmark_svm <- svm(gpw~Population, data=denmark_train, kernel='radial',
                   cost=0.1, gamma=0.5, epsilon=0.041)
mape(denmark_svm$fitted, denmark_train$gpw)
rmse(denmark_svm$fitted, denmark_train$gpw)
mae(denmark_svm$fitted, denmark_train$gpw)

denmark_svm_forecast <- predict(denmark_svm, newdata=denmark_test)
mape(denmark_svm_forecast, denmark_test$gpw)
rmse(denmark_svm_forecast, denmark_test$gpw)
mae(denmark_svm_forecast, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main='denmark gpw Recycling - SVM', col='black')
lines(y=denmark_svm$fitted, x=denmark_train$Year, col='red', type='b')
lines(y=denmark_svm_forecast, x=denmark_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(denmark_train, c('Population'))
proc.time()-start

set.seed(123)
denmark_rf <- randomForest(gpw~Population, data=denmark_train, maxnodes=6, ntree=500)

mape(denmark_rf$predicted, denmark_train$gpw)
rmse(denmark_rf$predicted, denmark_train$gpw)
mae(denmark_rf$predicted, denmark_train$gpw)

denmark_rf_forecast <- predict(denmark_rf, newdata=denmark_test)
mape(denmark_rf_forecast, denmark_test$gpw)
rmse(denmark_rf_forecast, denmark_test$gpw)
mae(denmark_rf_forecast, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main='denmark gpw Recycling - Radom Forest')
lines(y=denmark_rf$predicted, x=denmark_train$Year, type='b', col='red')
lines(y=denmark_rf_forecast, x=denmark_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(denmark_train, c('Population'))
proc.time()-start


denmark_X_train <- as.matrix(denmark_train[ , c('Population')])
denmark_y_train <- denmark_train$gpw
denmark_X_test  <- as.matrix(denmark_test[ , c('Population')])
denmark_y_test  <- denmark_test$gpw
denmark_dtrain <- xgb.DMatrix(data=denmark_X_train, label=denmark_y_train)
denmark_dtest  <- xgb.DMatrix(data=denmark_X_test)

set.seed(123)
denmark_xgb <- xgboost(data=denmark_dtrain, nrounds=100, max_depth=2,
                       eta=0.1, lambda=2, verbose=0)

denmark_xgb_fitted <- predict(denmark_xgb, denmark_dtrain)
mape(denmark_xgb_fitted, denmark_train$gpw)
rmse(denmark_xgb_fitted, denmark_train$gpw)
mae(denmark_xgb_fitted, denmark_train$gpw)

denmark_xgb_forecast <- predict(denmark_xgb, denmark_dtest)
mape(denmark_xgb_forecast, denmark_test$gpw)
rmse(denmark_xgb_forecast, denmark_test$gpw)
mae(denmark_xgb_forecast, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='gpw', main='denmark gpw - XGBoost')
lines(y=denmark_xgb_fitted, x=denmark_train$Year, type='b', col='red')
lines(y=denmark_xgb_forecast, x=denmark_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(denmark_train, c('Population'))
proc.time()-start

set.seed(123)
denmark_nnet <- nnetar(denmark_train$gpw, p=1, size=1, decay=0, 
                       repeats=20, xreg=as.matrix(denmark_train[, c('Population')]))

mape(as.numeric(fitted(denmark_nnet)[2:21]), denmark_train$gpw[2:21])
rmse(as.numeric(fitted(denmark_nnet)[2:21]), denmark_train$gpw[2:21])
mae(as.numeric(fitted(denmark_nnet)[2:21]), denmark_train$gpw[2:21])

denmark_nnet_forecast <- as.numeric(forecast(denmark_nnet, h=3, 
                                             xreg=as.matrix(denmark_test[, c('Population')]))$mean)

mape(denmark_nnet_forecast, denmark_test$gpw)
rmse(denmark_nnet_forecast, denmark_test$gpw)
mae(denmark_nnet_forecast, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main='denmark gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(denmark_nnet)), x=denmark_train$Year, type='b', col='red')
lines(y=denmark_nnet_forecast, x=denmark_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(denmark_train, c('Population'))
proc.time() - start

denmark_X_train <- as.matrix(denmark_train[, c('Population')])
denmark_X_test <- as.matrix(denmark_test[, c('Population')])

denmark_enet <- glmnet(denmark_X_train, denmark_train$gpw, alpha=0.6, lambda=2)

denmark_enet_fitted <- as.numeric(predict(denmark_enet, newx=denmark_X_train, s=2))
mape(denmark_enet_fitted, denmark_train$gpw)
rmse(denmark_enet_fitted, denmark_train$gpw)
mae(denmark_enet_fitted, denmark_train$gpw)

denmark_enet_forecast <- as.numeric(predict(denmark_enet, newx=denmark_X_test, s=2))
mape(denmark_enet_forecast, denmark_test$gpw)
rmse(denmark_enet_forecast, denmark_test$gpw)
mae(denmark_enet_forecast, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main='denmark gpw Recycling - Elastic Net', col='black')
lines(y=denmark_enet_fitted, x=denmark_train$Year, col='red', type='b')
lines(y=denmark_enet_forecast, x=denmark_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(denmark_train, c('Population'))
proc.time() - start

denmark_cubist <- cubist(x=as.matrix(denmark_train[, c('Population')]),
                         y=denmark_train$gpw, committees=10)

denmark_cubist_fitted <- predict(denmark_cubist, 
                                 newdata=as.data.frame(denmark_train[, c('Population')]),
                                 neighbors=1)
mape(denmark_cubist_fitted, denmark_train$gpw)
rmse(denmark_cubist_fitted, denmark_train$gpw)
mae(denmark_cubist_fitted, denmark_train$gpw)

denmark_cubist_forecast <- predict(denmark_cubist,
                                   newdata=as.data.frame(denmark_test[, c('Population')]),
                                   neighbors=1)
mape(denmark_cubist_forecast, denmark_test$gpw)
rmse(denmark_cubist_forecast, denmark_test$gpw)
mae(denmark_cubist_forecast, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main='denmark gpw Recycling - Cubist', col='black')
lines(y=denmark_cubist_fitted, x=denmark_train$Year, col='red', type='b')
lines(y=denmark_cubist_forecast, x=denmark_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
denmark_bayes_ridge <- bayesglm(gpw~Population, data=denmark_train,
                                family=gaussian)

denmark_bayes_fitted <- predict(denmark_bayes_ridge, newdata=denmark_train)
mape(denmark_bayes_fitted, denmark_train$gpw)
rmse(denmark_bayes_fitted, denmark_train$gpw)
mae(denmark_bayes_fitted, denmark_train$gpw)

denmark_bayes_forecast <- predict(denmark_bayes_ridge, newdata=denmark_test)
mape(denmark_bayes_forecast, denmark_test$gpw)
rmse(denmark_bayes_forecast, denmark_test$gpw)
mae(denmark_bayes_forecast, denmark_test$gpw)

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main='denmark gpw Recycling - Bayesian Ridge', col='black')
lines(y=denmark_bayes_fitted, x=denmark_train$Year, col='red', type='b')
lines(y=denmark_bayes_forecast, x=denmark_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#finland----
#train/test split (80/20)
finland_train <- finland[1:21, ]
finland_test <- finland[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(finland_train)


#xarima
finland_xarima <- auto.arima(y=finland_train$gpw, max.p=5, max.q=5,
                             xreg=as.matrix(finland_train[, c('Population')]))
mape(finland_xarima$fitted, finland_train$gpw)
rmse(finland_xarima$fitted, finland_train$gpw)
mae(finland_xarima$fitted, finland_train$gpw)

finland_xarima_forecast <- forecast(finland_xarima, h=nrow(finland_test),
                                    xreg=as.matrix(finland_test[, c('Population')]))

mape(finland_xarima_forecast$mean, finland_test$gpw)
rmse(finland_xarima_forecast$mean, finland_test$gpw)
mae(finland_xarima_forecast$mean, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main= 'finland gpw recycling rate - XARIMA', col='black')
lines(y=finland_xarima$fitted, x=finland_train$Year, col='red', type='b')
lines(y=finland_xarima_forecast$mean, x=finland_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~Population, 
                           data=finland_train)
finland_bsts <- bsts(gpw~Population,
                     state.specification=ss, data=finland_train, niter=1000)

finland_bsts_fitted <- predict(finland_bsts, newdata=finland_train,
                               horizon=nrow(finland_train), burn=100)
mape(finland_bsts_fitted$mean, finland_train$gpw)
rmse(finland_bsts_fitted$mean, finland_train$gpw)
mae(finland_bsts_fitted$mean, finland_train$gpw)

finland_bsts_forecast <- predict(finland_bsts, newdata=finland_test, burn=100)
mape(finland_bsts_forecast$mean, finland_test$gpw)
rmse(finland_bsts_forecast$mean, finland_test$gpw)
mae(finland_bsts_forecast$mean, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main='finland gpw Recycling - BSTS', col='black')
lines(y=finland_bsts_fitted$mean, x=finland_train$Year, col='red', type='b')
lines(y=finland_bsts_forecast$mean, x=finland_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(finland_train, c('Population'))
proc.time() - start

finland_svm <- svm(gpw~Population, data=finland_train, kernel='radial',
                   cost=4.6, gamma=0.1, epsilon=0.081)
mape(finland_svm$fitted, finland_train$gpw)
rmse(finland_svm$fitted, finland_train$gpw)
mae(finland_svm$fitted, finland_train$gpw)

finland_svm_forecast <- predict(finland_svm, newdata=finland_test)
mape(finland_svm_forecast, finland_test$gpw)
rmse(finland_svm_forecast, finland_test$gpw)
mae(finland_svm_forecast, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main='finland gpw Recycling - SVM', col='black')
lines(y=finland_svm$fitted, x=finland_train$Year, col='red', type='b')
lines(y=finland_svm_forecast, x=finland_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(finland_train, c('Population'))
proc.time()-start

set.seed(123)
finland_rf <- randomForest(gpw~Population, data=finland_train, maxnodes=6, ntree=400)

mape(finland_rf$predicted, finland_train$gpw)
rmse(finland_rf$predicted, finland_train$gpw)
mae(finland_rf$predicted, finland_train$gpw)

finland_rf_forecast <- predict(finland_rf, newdata=finland_test)
mape(finland_rf_forecast, finland_test$gpw)
rmse(finland_rf_forecast, finland_test$gpw)
mae(finland_rf_forecast, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main='finland gpw Recycling - Radom Forest')
lines(y=finland_rf$predicted, x=finland_train$Year, type='b', col='red')
lines(y=finland_rf_forecast, x=finland_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(finland_train, c('Population'))
proc.time()-start


finland_X_train <- as.matrix(finland_train[ , c('Population')])
finland_y_train <- finland_train$gpw
finland_X_test  <- as.matrix(finland_test[ , c('Population')])
finland_y_test  <- finland_test$gpw
finland_dtrain <- xgb.DMatrix(data=finland_X_train, label=finland_y_train)
finland_dtest  <- xgb.DMatrix(data=finland_X_test)

set.seed(123)
finland_xgb <- xgboost(data=finland_dtrain, nrounds=100, max_depth=2,
                       eta=0.2, lambda=10, verbose=0)

finland_xgb_fitted <- predict(finland_xgb, finland_dtrain)
mape(finland_xgb_fitted, finland_train$gpw)
rmse(finland_xgb_fitted, finland_train$gpw)
mae(finland_xgb_fitted, finland_train$gpw)

finland_xgb_forecast <- predict(finland_xgb, finland_dtest)
mape(finland_xgb_forecast, finland_test$gpw)
rmse(finland_xgb_forecast, finland_test$gpw)
mae(finland_xgb_forecast, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='gpw', main='finland gpw - XGBoost')
lines(y=finland_xgb_fitted, x=finland_train$Year, type='b', col='red')
lines(y=finland_xgb_forecast, x=finland_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(finland_train, c('Population'))
proc.time()-start

set.seed(123)
finland_nnet <- nnetar(finland_train$gpw, p=1, size=1, decay=0, 
                       repeats=10, xreg=as.matrix(finland_train[, c('Population')]))

mape(as.numeric(fitted(finland_nnet)[2:21]), finland_train$gpw[2:21])
rmse(as.numeric(fitted(finland_nnet)[2:21]), finland_train$gpw[2:21])
mae(as.numeric(fitted(finland_nnet)[2:21]), finland_train$gpw[2:21])

finland_nnet_forecast <- as.numeric(forecast(finland_nnet, h=3, 
                                             xreg=as.matrix(finland_test[, c('Population')]))$mean)

mape(finland_nnet_forecast, finland_test$gpw)
rmse(finland_nnet_forecast, finland_test$gpw)
mae(finland_nnet_forecast, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main='finland gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(finland_nnet)), x=finland_train$Year, type='b', col='red')
lines(y=finland_nnet_forecast, x=finland_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(finland_train, c('Population'))
proc.time() - start

finland_X_train <- as.matrix(finland_train[, c('Population')])
finland_X_test <- as.matrix(finland_test[, c('Population')])

finland_enet <- glmnet(finland_X_train, finland_train$gpw, alpha=0.6, lambda=2)

finland_enet_fitted <- as.numeric(predict(finland_enet, newx=finland_X_train, s=2))
mape(finland_enet_fitted, finland_train$gpw)
rmse(finland_enet_fitted, finland_train$gpw)
mae(finland_enet_fitted, finland_train$gpw)

finland_enet_forecast <- as.numeric(predict(finland_enet, newx=finland_X_test, s=2))
mape(finland_enet_forecast, finland_test$gpw)
rmse(finland_enet_forecast, finland_test$gpw)
mae(finland_enet_forecast, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main='finland gpw Recycling - Elastic Net', col='black')
lines(y=finland_enet_fitted, x=finland_train$Year, col='red', type='b')
lines(y=finland_enet_forecast, x=finland_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(finland_train, c('Population'))
proc.time() - start

finland_cubist <- cubist(x=as.matrix(finland_train[, c('Population')]),
                         y=finland_train$gpw, committees=5)

finland_cubist_fitted <- predict(finland_cubist, 
                                 newdata=as.data.frame(finland_train[, c('Population')]),
                                 neighbors=5)
mape(finland_cubist_fitted, finland_train$gpw)
rmse(finland_cubist_fitted, finland_train$gpw)
mae(finland_cubist_fitted, finland_train$gpw)

finland_cubist_forecast <- predict(finland_cubist,
                                   newdata=as.data.frame(finland_test[, c('Population')]),
                                   neighbors=5)
mape(finland_cubist_forecast, finland_test$gpw)
rmse(finland_cubist_forecast, finland_test$gpw)
mae(finland_cubist_forecast, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main='finland gpw Recycling - Cubist', col='black')
lines(y=finland_cubist_fitted, x=finland_train$Year, col='red', type='b')
lines(y=finland_cubist_forecast, x=finland_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
finland_bayes_ridge <- bayesglm(gpw~Population, data=finland_train,
                                family=gaussian)

finland_bayes_fitted <- predict(finland_bayes_ridge, newdata=finland_train)
mape(finland_bayes_fitted, finland_train$gpw)
rmse(finland_bayes_fitted, finland_train$gpw)
mae(finland_bayes_fitted, finland_train$gpw)

finland_bayes_forecast <- predict(finland_bayes_ridge, newdata=finland_test)
mape(finland_bayes_forecast, finland_test$gpw)
rmse(finland_bayes_forecast, finland_test$gpw)
mae(finland_bayes_forecast, finland_test$gpw)

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main='finland gpw Recycling - Bayesian Ridge', col='black')
lines(y=finland_bayes_fitted, x=finland_train$Year, col='red', type='b')
lines(y=finland_bayes_forecast, x=finland_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#france----
#train/test split (80/20)
france_train <- france[1:21, ]
france_test <- france[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(france_train)


#xarima
france_xarima <- auto.arima(y=france_train$gpw, max.p=5, max.q=5,
                            xreg=as.matrix(france_train[, c('Population')]))
mape(france_xarima$fitted, france_train$gpw)
rmse(france_xarima$fitted, france_train$gpw)
mae(france_xarima$fitted, france_train$gpw)

france_xarima_forecast <- forecast(france_xarima, h=nrow(france_test),
                                   xreg=as.matrix(france_test[, c('Population')]))

mape(france_xarima_forecast$mean, france_test$gpw)
rmse(france_xarima_forecast$mean, france_test$gpw)
mae(france_xarima_forecast$mean, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main= 'france gpw recycling rate - XARIMA', col='black')
lines(y=france_xarima$fitted, x=france_train$Year, col='red', type='b')
lines(y=france_xarima_forecast$mean, x=france_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~Population, 
                           data=france_train)
france_bsts <- bsts(gpw~Population,
                    state.specification=ss, data=france_train, niter=1000)

france_bsts_fitted <- predict(france_bsts, newdata=france_train,
                              horizon=nrow(france_train), burn=100)
mape(france_bsts_fitted$mean, france_train$gpw)
rmse(france_bsts_fitted$mean, france_train$gpw)
mae(france_bsts_fitted$mean, france_train$gpw)

france_bsts_forecast <- predict(france_bsts, newdata=france_test, burn=100)
mape(france_bsts_forecast$mean, france_test$gpw)
rmse(france_bsts_forecast$mean, france_test$gpw)
mae(france_bsts_forecast$mean, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main='france gpw Recycling - BSTS', col='black')
lines(y=france_bsts_fitted$mean, x=france_train$Year, col='red', type='b')
lines(y=france_bsts_forecast$mean, x=france_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(france_train, c('Population'))
proc.time() - start

france_svm <- svm(gpw~Population, data=france_train, kernel='radial',
                  cost=5, gamma=0.45, epsilon=0.081)
mape(france_svm$fitted, france_train$gpw)
rmse(france_svm$fitted, france_train$gpw)
mae(france_svm$fitted, france_train$gpw)

france_svm_forecast <- predict(france_svm, newdata=france_test)
mape(france_svm_forecast, france_test$gpw)
rmse(france_svm_forecast, france_test$gpw)
mae(france_svm_forecast, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main='france gpw Recycling - SVM', col='black')
lines(y=france_svm$fitted, x=france_train$Year, col='red', type='b')
lines(y=france_svm_forecast, x=france_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(france_train, c('Population'))
proc.time()-start

set.seed(123)
france_rf <- randomForest(gpw~Population, data=france_train, maxnodes=10, ntree=50)

mape(france_rf$predicted, france_train$gpw)
rmse(france_rf$predicted, france_train$gpw)
mae(france_rf$predicted, france_train$gpw)

france_rf_forecast <- predict(france_rf, newdata=france_test)
mape(france_rf_forecast, france_test$gpw)
rmse(france_rf_forecast, france_test$gpw)
mae(france_rf_forecast, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main='france gpw Recycling - Radom Forest')
lines(y=france_rf$predicted, x=france_train$Year, type='b', col='red')
lines(y=france_rf_forecast, x=france_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(france_train, c('Population'))
proc.time()-start

france_X_train <- as.matrix(france_train[ , c('Population')])
france_y_train <- france_train$gpw
france_X_test  <- as.matrix(france_test[ , c('Population')])
france_y_test  <- france_test$gpw
france_dtrain <- xgb.DMatrix(data=france_X_train, label=france_y_train)
france_dtest  <- xgb.DMatrix(data=france_X_test)

set.seed(123)
france_xgb <- xgboost(data=france_dtrain, nrounds=100, max_depth=3,
                      eta=0.3, lambda=0.1, verbose=0)

france_xgb_fitted <- predict(france_xgb, france_dtrain)
mape(france_xgb_fitted, france_train$gpw)
rmse(france_xgb_fitted, france_train$gpw)
mae(france_xgb_fitted, france_train$gpw)

france_xgb_forecast <- predict(france_xgb, france_dtest)
mape(france_xgb_forecast, france_test$gpw)
rmse(france_xgb_forecast, france_test$gpw)
mae(france_xgb_forecast, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='gpw', main='france gpw - XGBoost')
lines(y=france_xgb_fitted, x=france_train$Year, type='b', col='red')
lines(y=france_xgb_forecast, x=france_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(france_train, c('Population'))
proc.time()-start

set.seed(123)
france_nnet <- nnetar(france_train$gpw, p=1, size=3, decay=0, 
                       repeats=50, xreg=as.matrix(france_train[, c('Population')]))

mape(as.numeric(fitted(france_nnet)[2:21]), france_train$gpw[2:21])
rmse(as.numeric(fitted(france_nnet)[2:21]), france_train$gpw[2:21])
mae(as.numeric(fitted(france_nnet)[2:21]), france_train$gpw[2:21])

france_nnet_forecast <- as.numeric(forecast(france_nnet, h=3, 
                                             xreg=as.matrix(france_test[, c('Population')]))$mean)

mape(france_nnet_forecast, france_test$gpw)
rmse(france_nnet_forecast, france_test$gpw)
mae(france_nnet_forecast, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main='france gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(france_nnet)), x=france_train$Year, type='b', col='red')
lines(y=france_nnet_forecast, x=france_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(france_train, c('Population'))
proc.time() - start

france_X_train <- as.matrix(france_train[, c('Population')])
france_X_test <- as.matrix(france_test[, c('Population')])

france_enet <- glmnet(france_X_train, france_train$gpw, alpha=0.6, lambda=2)

france_enet_fitted <- as.numeric(predict(france_enet, newx=france_X_train, s=2))
mape(france_enet_fitted, france_train$gpw)
rmse(france_enet_fitted, france_train$gpw)
mae(france_enet_fitted, france_train$gpw)

france_enet_forecast <- as.numeric(predict(france_enet, newx=france_X_test, s=2))
mape(france_enet_forecast, france_test$gpw)
rmse(france_enet_forecast, france_test$gpw)
mae(france_enet_forecast, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main='france gpw Recycling - Elastic Net', col='black')
lines(y=france_enet_fitted, x=france_train$Year, col='red', type='b')
lines(y=france_enet_forecast, x=france_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(france_train, c('Population'))
proc.time() - start

france_cubist <- cubist(x=as.matrix(france_train[, c('Population')]),
                         y=france_train$gpw, committees=5)

france_cubist_fitted <- predict(france_cubist, 
                                 newdata=as.data.frame(france_train[, c('Population')]),
                                 neighbors=9)
mape(france_cubist_fitted, france_train$gpw)
rmse(france_cubist_fitted, france_train$gpw)
mae(france_cubist_fitted, france_train$gpw)

france_cubist_forecast <- predict(france_cubist,
                                   newdata=as.data.frame(france_test[, c('Population')]),
                                   neighbors=9)
mape(france_cubist_forecast, france_test$gpw)
rmse(france_cubist_forecast, france_test$gpw)
mae(france_cubist_forecast, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main='france gpw Recycling - Cubist', col='black')
lines(y=france_cubist_fitted, x=france_train$Year, col='red', type='b')
lines(y=france_cubist_forecast, x=france_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
france_bayes_ridge <- bayesglm(gpw~Population, data=france_train,
                                family=gaussian)

france_bayes_fitted <- predict(france_bayes_ridge, newdata=france_train)
mape(france_bayes_fitted, france_train$gpw)
rmse(france_bayes_fitted, france_train$gpw)
mae(france_bayes_fitted, france_train$gpw)

france_bayes_forecast <- predict(france_bayes_ridge, newdata=france_test)
mape(france_bayes_forecast, france_test$gpw)
rmse(france_bayes_forecast, france_test$gpw)
mae(france_bayes_forecast, france_test$gpw)

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main='france gpw Recycling - Bayesian Ridge', col='black')
lines(y=france_bayes_fitted, x=france_train$Year, col='red', type='b')
lines(y=france_bayes_forecast, x=france_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#germany----
#train/test split (80/20)
germany_train <- germany[1:21, ]
germany_test <- germany[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(germany_train)


#xarima
germany_xarima <- auto.arima(y=germany_train$gpw, max.p=5, max.q=5,
                             xreg=as.matrix(germany_train[, c('Population')]))
mape(germany_xarima$fitted, germany_train$gpw)
rmse(germany_xarima$fitted, germany_train$gpw)
mae(germany_xarima$fitted, germany_train$gpw)

germany_xarima_forecast <- forecast(germany_xarima, h=nrow(germany_test),
                                    xreg=as.matrix(germany_test[, c('Population')]))

mape(germany_xarima_forecast$mean, germany_test$gpw)
rmse(germany_xarima_forecast$mean, germany_test$gpw)
mae(germany_xarima_forecast$mean, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main= 'germany gpw recycling rate - XARIMA', col='black')
lines(y=germany_xarima$fitted, x=germany_train$Year, col='red', type='b')
lines(y=germany_xarima_forecast$mean, x=germany_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~Population, 
                           data=germany_train)
germany_bsts <- bsts(gpw~Population,
                     state.specification=ss, data=germany_train, niter=1000)

germany_bsts_fitted <- predict(germany_bsts, newdata=germany_train,
                               horizon=nrow(germany_train), burn=100)
mape(germany_bsts_fitted$mean, germany_train$gpw)
rmse(germany_bsts_fitted$mean, germany_train$gpw)
mae(germany_bsts_fitted$mean, germany_train$gpw)

germany_bsts_forecast <- predict(germany_bsts, newdata=germany_test, burn=100)
mape(germany_bsts_forecast$mean, germany_test$gpw)
rmse(germany_bsts_forecast$mean, germany_test$gpw)
mae(germany_bsts_forecast$mean, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main='germany gpw Recycling - BSTS', col='black')
lines(y=germany_bsts_fitted$mean, x=germany_train$Year, col='red', type='b')
lines(y=germany_bsts_forecast$mean, x=germany_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(germany_train, c('Population'))
proc.time() - start

germany_svm <- svm(gpw~Population, data=germany_train, kernel='radial',
                   cost=0.1, gamma=0.25, epsilon=0.091)
mape(germany_svm$fitted, germany_train$gpw)
rmse(germany_svm$fitted, germany_train$gpw)
mae(germany_svm$fitted, germany_train$gpw)

germany_svm_forecast <- predict(germany_svm, newdata=germany_test)
mape(germany_svm_forecast, germany_test$gpw)
rmse(germany_svm_forecast, germany_test$gpw)
mae(germany_svm_forecast, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main='germany gpw Recycling - SVM', col='black')
lines(y=germany_svm$fitted, x=germany_train$Year, col='red', type='b')
lines(y=germany_svm_forecast, x=germany_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(germany_train, c('Population'))
proc.time()-start

set.seed(123)
germany_rf <- randomForest(gpw~Population, data=germany_train, maxnodes=1, ntree=100)

mape(germany_rf$predicted, germany_train$gpw)
rmse(germany_rf$predicted, germany_train$gpw)
mae(germany_rf$predicted, germany_train$gpw)

germany_rf_forecast <- predict(germany_rf, newdata=germany_test)
mape(germany_rf_forecast, germany_test$gpw)
rmse(germany_rf_forecast, germany_test$gpw)
mae(germany_rf_forecast, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main='germany gpw Recycling - Radom Forest')
lines(y=germany_rf$predicted, x=germany_train$Year, type='b', col='red')
lines(y=germany_rf_forecast, x=germany_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(germany_train, c('Population'))
proc.time()-start

germany_X_train <- as.matrix(germany_train[ , c('Population')])
germany_y_train <- germany_train$gpw
germany_X_test  <- as.matrix(germany_test[ , c('Population')])
germany_y_test  <- germany_test$gpw
germany_dtrain <- xgb.DMatrix(data=germany_X_train, label=germany_y_train)
germany_dtest  <- xgb.DMatrix(data=germany_X_test)

set.seed(123)
germany_xgb <- xgboost(data=germany_dtrain, nrounds=100, max_depth=2,
                       eta=0.1, lambda=10, verbose=0)

germany_xgb_fitted <- predict(germany_xgb, germany_dtrain)
mape(germany_xgb_fitted, germany_train$gpw)
rmse(germany_xgb_fitted, germany_train$gpw)
mae(germany_xgb_fitted, germany_train$gpw)

germany_xgb_forecast <- predict(germany_xgb, germany_dtest)
mape(germany_xgb_forecast, germany_test$gpw)
rmse(germany_xgb_forecast, germany_test$gpw)
mae(germany_xgb_forecast, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='gpw', main='germany gpw - XGBoost')
lines(y=germany_xgb_fitted, x=germany_train$Year, type='b', col='red')
lines(y=germany_xgb_forecast, x=germany_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(germany_train, c('Population'))
proc.time()-start

set.seed(123)
germany_nnet <- nnetar(germany_train$gpw, p=1, size=1, decay=0.1, 
                       repeats=20, xreg=as.matrix(germany_train[, c('Population')]))

mape(as.numeric(fitted(germany_nnet)[2:21]), germany_train$gpw[2:21])
rmse(as.numeric(fitted(germany_nnet)[2:21]), germany_train$gpw[2:21])
mae(as.numeric(fitted(germany_nnet)[2:21]), germany_train$gpw[2:21])

germany_nnet_forecast <- as.numeric(forecast(germany_nnet, h=3, 
                                             xreg=as.matrix(germany_test[, c('Population')]))$mean)

mape(germany_nnet_forecast, germany_test$gpw)
rmse(germany_nnet_forecast, germany_test$gpw)
mae(germany_nnet_forecast, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main='germany gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(germany_nnet)), x=germany_train$Year, type='b', col='red')
lines(y=germany_nnet_forecast, x=germany_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#elastic net
start <- proc.time()
rolling_cv_enet(germany_train, c('Population'))
proc.time() - start

germany_X_train <- as.matrix(germany_train[, c('Population')])
germany_X_test <- as.matrix(germany_test[, c('Population')])

germany_enet <- glmnet(germany_X_train, germany_train$gpw, alpha=0.6, lambda=2)

germany_enet_fitted <- as.numeric(predict(germany_enet, newx=germany_X_train, s=2))
mape(germany_enet_fitted, germany_train$gpw)
rmse(germany_enet_fitted, germany_train$gpw)
mae(germany_enet_fitted, germany_train$gpw)

germany_enet_forecast <- as.numeric(predict(germany_enet, newx=germany_X_test, s=2))
mape(germany_enet_forecast, germany_test$gpw)
rmse(germany_enet_forecast, germany_test$gpw)
mae(germany_enet_forecast, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main='germany gpw Recycling - Elastic Net', col='black')
lines(y=germany_enet_fitted, x=germany_train$Year, col='red', type='b')
lines(y=germany_enet_forecast, x=germany_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(germany_train, c('Population'))
proc.time() - start

germany_cubist <- cubist(x=as.matrix(germany_train[, c('Population')]),
                        y=germany_train$gpw, committees=1)

germany_cubist_fitted <- predict(germany_cubist, 
                                newdata=as.data.frame(germany_train[, c('Population')]),
                                neighbors=3)
mape(germany_cubist_fitted, germany_train$gpw)
rmse(germany_cubist_fitted, germany_train$gpw)
mae(germany_cubist_fitted, germany_train$gpw)

germany_cubist_forecast <- predict(germany_cubist,
                                  newdata=as.data.frame(germany_test[, c('Population')]),
                                  neighbors=3)
mape(germany_cubist_forecast, germany_test$gpw)
rmse(germany_cubist_forecast, germany_test$gpw)
mae(germany_cubist_forecast, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main='germany gpw Recycling - Cubist', col='black')
lines(y=germany_cubist_fitted, x=germany_train$Year, col='red', type='b')
lines(y=germany_cubist_forecast, x=germany_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
germany_bayes_ridge <- bayesglm(gpw~Population, data=germany_train,
                               family=gaussian)

germany_bayes_fitted <- predict(germany_bayes_ridge, newdata=germany_train)
mape(germany_bayes_fitted, germany_train$gpw)
rmse(germany_bayes_fitted, germany_train$gpw)
mae(germany_bayes_fitted, germany_train$gpw)

germany_bayes_forecast <- predict(germany_bayes_ridge, newdata=germany_test)
mape(germany_bayes_forecast, germany_test$gpw)
rmse(germany_bayes_forecast, germany_test$gpw)
mae(germany_bayes_forecast, germany_test$gpw)

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main='germany gpw Recycling - Bayesian Ridge', col='black')
lines(y=germany_bayes_fitted, x=germany_train$Year, col='red', type='b')
lines(y=germany_bayes_forecast, x=germany_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#ireland----
#train/test split (80/20)
ireland_train <- ireland[1:21, ]
ireland_test <- ireland[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(ireland_train)


#xarima
ireland_xarima <- auto.arima(y=ireland_train$gpw, max.p=5, max.q=5,
                             xreg=as.matrix(ireland_train[, c('Population')]))
mape(ireland_xarima$fitted, ireland_train$gpw)
rmse(ireland_xarima$fitted, ireland_train$gpw)
mae(ireland_xarima$fitted, ireland_train$gpw)

ireland_xarima_forecast <- forecast(ireland_xarima, h=nrow(ireland_test),
                                    xreg=as.matrix(ireland_test[, c('Population')]))

mape(ireland_xarima_forecast$mean, ireland_test$gpw)
rmse(ireland_xarima_forecast$mean, ireland_test$gpw)
mae(ireland_xarima_forecast$mean, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main= 'ireland gpw recycling rate - XARIMA', col='black')
lines(y=ireland_xarima$fitted, x=ireland_train$Year, col='red', type='b')
lines(y=ireland_xarima_forecast$mean, x=ireland_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~Population, 
                           data=ireland_train)
ireland_bsts <- bsts(gpw~Population,
                     state.specification=ss, data=ireland_train, niter=1000)

ireland_bsts_fitted <- predict(ireland_bsts, newdata=ireland_train,
                               horizon=nrow(ireland_train), burn=100)
mape(ireland_bsts_fitted$mean, ireland_train$gpw)
rmse(ireland_bsts_fitted$mean, ireland_train$gpw)
mae(ireland_bsts_fitted$mean, ireland_train$gpw)

ireland_bsts_forecast <- predict(ireland_bsts, newdata=ireland_test, burn=100)
mape(ireland_bsts_forecast$mean, ireland_test$gpw)
rmse(ireland_bsts_forecast$mean, ireland_test$gpw)
mae(ireland_bsts_forecast$mean, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main='ireland gpw Recycling - BSTS', col='black')
lines(y=ireland_bsts_fitted$mean, x=ireland_train$Year, col='red', type='b')
lines(y=ireland_bsts_forecast$mean, x=ireland_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(ireland_train, c('Population'))
proc.time() - start

ireland_svm <- svm(gpw~Population, data=ireland_train, kernel='radial',
                   cost=3.9, gamma=0.05, epsilon=0.081)
mape(ireland_svm$fitted, ireland_train$gpw)
rmse(ireland_svm$fitted, ireland_train$gpw)
mae(ireland_svm$fitted, ireland_train$gpw)

ireland_svm_forecast <- predict(ireland_svm, newdata=ireland_test)
mape(ireland_svm_forecast, ireland_test$gpw)
rmse(ireland_svm_forecast, ireland_test$gpw)
mae(ireland_svm_forecast, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main='ireland gpw Recycling - SVM', col='black')
lines(y=ireland_svm$fitted, x=ireland_train$Year, col='red', type='b')
lines(y=ireland_svm_forecast, x=ireland_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(ireland_train, c('Population'))
proc.time()-start

set.seed(123)
ireland_rf <- randomForest(gpw~Population, data=ireland_train, maxnodes=7, ntree=400)

mape(ireland_rf$predicted, ireland_train$gpw)
rmse(ireland_rf$predicted, ireland_train$gpw)
mae(ireland_rf$predicted, ireland_train$gpw)

ireland_rf_forecast <- predict(ireland_rf, newdata=ireland_test)
mape(ireland_rf_forecast, ireland_test$gpw)
rmse(ireland_rf_forecast, ireland_test$gpw)
mae(ireland_rf_forecast, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main='ireland gpw Recycling - Radom Forest')
lines(y=ireland_rf$predicted, x=ireland_train$Year, type='b', col='red')
lines(y=ireland_rf_forecast, x=ireland_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(ireland_train, c('Population'))
proc.time()-start

ireland_X_train <- as.matrix(ireland_train[ , c('Population')])
ireland_y_train <- ireland_train$gpw
ireland_X_test  <- as.matrix(ireland_test[ , c('Population')])
ireland_y_test  <- ireland_test$gpw
ireland_dtrain <- xgb.DMatrix(data=ireland_X_train, label=ireland_y_train)
ireland_dtest  <- xgb.DMatrix(data=ireland_X_test)

set.seed(123)
ireland_xgb <- xgboost(data=ireland_dtrain, nrounds=100, max_depth=2,
                       eta=0.05, lambda=0.1, verbose=0)

ireland_xgb_fitted <- predict(ireland_xgb, ireland_dtrain)
mape(ireland_xgb_fitted, ireland_train$gpw)
rmse(ireland_xgb_fitted, ireland_train$gpw)
mae(ireland_xgb_fitted, ireland_train$gpw)

ireland_xgb_forecast <- predict(ireland_xgb, ireland_dtest)
mape(ireland_xgb_forecast, ireland_test$gpw)
rmse(ireland_xgb_forecast, ireland_test$gpw)
mae(ireland_xgb_forecast, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='gpw', main='ireland gpw - XGBoost')
lines(y=ireland_xgb_fitted, x=ireland_train$Year, type='b', col='red')
lines(y=ireland_xgb_forecast, x=ireland_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(ireland_train, c('Population'))
proc.time()-start

set.seed(123)
ireland_nnet <- nnetar(ireland_train$gpw, p=1, size=4, decay=0.1, 
                       repeats=20, xreg=as.matrix(ireland_train[, c('Population')]))

mape(as.numeric(fitted(ireland_nnet)[2:21]), ireland_train$gpw[2:21])
rmse(as.numeric(fitted(ireland_nnet)[2:21]), ireland_train$gpw[2:21])
mae(as.numeric(fitted(ireland_nnet)[2:21]), ireland_train$gpw[2:21])

ireland_nnet_forecast <- as.numeric(forecast(ireland_nnet, h=3, 
                                             xreg=as.matrix(ireland_test[, c('Population')]))$mean)

mape(ireland_nnet_forecast, ireland_test$gpw)
rmse(ireland_nnet_forecast, ireland_test$gpw)
mae(ireland_nnet_forecast, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main='ireland gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(ireland_nnet)), x=ireland_train$Year, type='b', col='red')
lines(y=ireland_nnet_forecast, x=ireland_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(ireland_train, c('Population'))
proc.time() - start

ireland_X_train <- as.matrix(ireland_train[, c('Population')])
ireland_X_test <- as.matrix(ireland_test[, c('Population')])

ireland_enet <- glmnet(ireland_X_train, ireland_train$gpw, alpha=0.6, lambda=2)

ireland_enet_fitted <- as.numeric(predict(ireland_enet, newx=ireland_X_train, s=2))
mape(ireland_enet_fitted, ireland_train$gpw)
rmse(ireland_enet_fitted, ireland_train$gpw)
mae(ireland_enet_fitted, ireland_train$gpw)

ireland_enet_forecast <- as.numeric(predict(ireland_enet, newx=ireland_X_test, s=2))
mape(ireland_enet_forecast, ireland_test$gpw)
rmse(ireland_enet_forecast, ireland_test$gpw)
mae(ireland_enet_forecast, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main='ireland gpw Recycling - Elastic Net', col='black')
lines(y=ireland_enet_fitted, x=ireland_train$Year, col='red', type='b')
lines(y=ireland_enet_forecast, x=ireland_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(ireland_train, c('Population'))
proc.time() - start

ireland_cubist <- cubist(x=as.matrix(ireland_train[, c('Population')]),
                         y=ireland_train$gpw, committees=5)

ireland_cubist_fitted <- predict(ireland_cubist, 
                                 newdata=as.data.frame(ireland_train[, c('Population')]),
                                 neighbors=7)
mape(ireland_cubist_fitted, ireland_train$gpw)
rmse(ireland_cubist_fitted, ireland_train$gpw)
mae(ireland_cubist_fitted, ireland_train$gpw)

ireland_cubist_forecast <- predict(ireland_cubist,
                                   newdata=as.data.frame(ireland_test[, c('Population')]),
                                   neighbors=7)
mape(ireland_cubist_forecast, ireland_test$gpw)
rmse(ireland_cubist_forecast, ireland_test$gpw)
mae(ireland_cubist_forecast, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main='ireland gpw Recycling - Cubist', col='black')
lines(y=ireland_cubist_fitted, x=ireland_train$Year, col='red', type='b')
lines(y=ireland_cubist_forecast, x=ireland_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
ireland_bayes_ridge <- bayesglm(gpw~Population, data=ireland_train,
                                family=gaussian)

ireland_bayes_fitted <- predict(ireland_bayes_ridge, newdata=ireland_train)
mape(ireland_bayes_fitted, ireland_train$gpw)
rmse(ireland_bayes_fitted, ireland_train$gpw)
mae(ireland_bayes_fitted, ireland_train$gpw)

ireland_bayes_forecast <- predict(ireland_bayes_ridge, newdata=ireland_test)
mape(ireland_bayes_forecast, ireland_test$gpw)
rmse(ireland_bayes_forecast, ireland_test$gpw)
mae(ireland_bayes_forecast, ireland_test$gpw)

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main='ireland gpw Recycling - Bayesian Ridge', col='black')
lines(y=ireland_bayes_fitted, x=ireland_train$Year, col='red', type='b')
lines(y=ireland_bayes_forecast, x=ireland_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#italy----
#train/test split (80/20)
italy_train <- italy[1:21, ]
italy_test <- italy[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(italy_train)


#xarima
italy_xarima <- auto.arima(y=italy_train$gpw, max.p=5, max.q=5,
                           xreg=as.matrix(italy_train[, c('Population', 'GDP', 'co2', 'energy_consumption')]))
mape(italy_xarima$fitted, italy_train$gpw)
rmse(italy_xarima$fitted, italy_train$gpw)
mae(italy_xarima$fitted, italy_train$gpw)

italy_xarima_forecast <- forecast(italy_xarima, h=nrow(italy_test),
                                  xreg=as.matrix(italy_test[, c('Population', 'GDP', 'co2', 'energy_consumption')]))

mape(italy_xarima_forecast$mean, italy_test$gpw)
rmse(italy_xarima_forecast$mean, italy_test$gpw)
mae(italy_xarima_forecast$mean, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main= 'italy gpw recycling rate - XARIMA', col='black')
lines(y=italy_xarima$fitted, x=italy_train$Year, col='red', type='b')
lines(y=italy_xarima_forecast$mean, x=italy_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~Population+GDP+co2+energy_consumption, 
                           data=italy_train)
italy_bsts <- bsts(gpw~Population+GDP+co2+energy_consumption,
                   state.specification=ss, data=italy_train, niter=1000)

italy_bsts_fitted <- predict(italy_bsts, newdata=italy_train,
                             horizon=nrow(italy_train), burn=100)
mape(italy_bsts_fitted$mean, italy_train$gpw)
rmse(italy_bsts_fitted$mean, italy_train$gpw)
mae(italy_bsts_fitted$mean, italy_train$gpw)

italy_bsts_forecast <- predict(italy_bsts, newdata=italy_test, burn=100)
mape(italy_bsts_forecast$mean, italy_test$gpw)
rmse(italy_bsts_forecast$mean, italy_test$gpw)
mae(italy_bsts_forecast$mean, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main='italy gpw Recycling - BSTS', col='black')
lines(y=italy_bsts_fitted$mean, x=italy_train$Year, col='red', type='b')
lines(y=italy_bsts_forecast$mean, x=italy_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(italy_train, c('Population', 'GDP', 'co2', 'energy_consumption'))
proc.time() - start

italy_svm <- svm(gpw~Population+GDP+co2+energy_consumption, data=italy_train, kernel='radial',
                 cost=3.1, gamma=0.05, epsilon=0.021)
mape(italy_svm$fitted, italy_train$gpw)
rmse(italy_svm$fitted, italy_train$gpw)
mae(italy_svm$fitted, italy_train$gpw)

italy_svm_forecast <- predict(italy_svm, newdata=italy_test)
mape(italy_svm_forecast, italy_test$gpw)
rmse(italy_svm_forecast, italy_test$gpw)
mae(italy_svm_forecast, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main='italy gpw Recycling - SVM', col='black')
lines(y=italy_svm$fitted, x=italy_train$Year, col='red', type='b')
lines(y=italy_svm_forecast, x=italy_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(italy_train, c('Population', 'GDP', 'co2', 'energy_consumption'))
proc.time()-start

set.seed(123)
italy_rf <- randomForest(gpw~Population+GDP+co2+energy_consumption, data=italy_train, maxnodes=10, ntree=50)

mape(italy_rf$predicted, italy_train$gpw)
rmse(italy_rf$predicted, italy_train$gpw)
mae(italy_rf$predicted, italy_train$gpw)

italy_rf_forecast <- predict(italy_rf, newdata=italy_test)
mape(italy_rf_forecast, italy_test$gpw)
rmse(italy_rf_forecast, italy_test$gpw)
mae(italy_rf_forecast, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main='italy gpw Recycling - Radom Forest')
lines(y=italy_rf$predicted, x=italy_train$Year, type='b', col='red')
lines(y=italy_rf_forecast, x=italy_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(italy_train, c('Population', 'GDP', 'co2', 'energy_consumption'))
proc.time()-start

italy_X_train <- as.matrix(italy_train[ , c('Population', 'GDP', 'co2', 'energy_consumption')])
italy_y_train <- italy_train$gpw
italy_X_test  <- as.matrix(italy_test[ , c('Population', 'GDP', 'co2', 'energy_consumption')])
italy_y_test  <- italy_test$gpw
italy_dtrain <- xgb.DMatrix(data=italy_X_train, label=italy_y_train)
italy_dtest  <- xgb.DMatrix(data=italy_X_test)

set.seed(123)
italy_xgb <- xgboost(data=italy_dtrain, nrounds=100, max_depth=3,
                     eta=0.3, lambda=0.1, verbose=0)

italy_xgb_fitted <- predict(italy_xgb, italy_dtrain)
mape(italy_xgb_fitted, italy_train$gpw)
rmse(italy_xgb_fitted, italy_train$gpw)
mae(italy_xgb_fitted, italy_train$gpw)

italy_xgb_forecast <- predict(italy_xgb, italy_dtest)
mape(italy_xgb_forecast, italy_test$gpw)
rmse(italy_xgb_forecast, italy_test$gpw)
mae(italy_xgb_forecast, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='gpw', main='italy gpw - XGBoost')
lines(y=italy_xgb_fitted, x=italy_train$Year, type='b', col='red')
lines(y=italy_xgb_forecast, x=italy_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(italy_train, c('Population', 'GDP', 'co2', 'energy_consumption'))
proc.time()-start

set.seed(123)
italy_nnet <- nnetar(italy_train$gpw, p=1, size=5, decay=0.1, 
                       repeats=50, xreg=as.matrix(italy_train[, c('Population', 'GDP', 'co2', 'energy_consumption')]))

mape(as.numeric(fitted(italy_nnet)[2:21]), italy_train$gpw[2:21])
rmse(as.numeric(fitted(italy_nnet)[2:21]), italy_train$gpw[2:21])
mae(as.numeric(fitted(italy_nnet)[2:21]), italy_train$gpw[2:21])

italy_nnet_forecast <- as.numeric(forecast(italy_nnet, h=3, 
                                             xreg=as.matrix(italy_test[, c('Population', 'GDP', 'co2', 'energy_consumption')]))$mean)

mape(italy_nnet_forecast, italy_test$gpw)
rmse(italy_nnet_forecast, italy_test$gpw)
mae(italy_nnet_forecast, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main='italy gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(italy_nnet)), x=italy_train$Year, type='b', col='red')
lines(y=italy_nnet_forecast, x=italy_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(italy_train, c('Population', 'GDP', 'co2', 'energy_consumption'))
proc.time() - start

italy_X_train <- as.matrix(italy_train[, c('Population', 'GDP', 'co2', 'energy_consumption')])
italy_X_test <- as.matrix(italy_test[, c('Population', 'GDP', 'co2', 'energy_consumption')])

italy_enet <- glmnet(italy_X_train, italy_train$gpw, alpha=0.5, lambda=0.5)

italy_enet_fitted <- as.numeric(predict(italy_enet, newx=italy_X_train, s=0.5))
mape(italy_enet_fitted, italy_train$gpw)
rmse(italy_enet_fitted, italy_train$gpw)
mae(italy_enet_fitted, italy_train$gpw)

italy_enet_forecast <- as.numeric(predict(italy_enet, newx=italy_X_test, s=0.5))
mape(italy_enet_forecast, italy_test$gpw)
rmse(italy_enet_forecast, italy_test$gpw)
mae(italy_enet_forecast, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main='italy gpw Recycling - Elastic Net', col='black')
lines(y=italy_enet_fitted, x=italy_train$Year, col='red', type='b')
lines(y=italy_enet_forecast, x=italy_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(italy_train, c('Population', 'GDP', 'co2', 'energy_consumption'))
proc.time() - start

italy_cubist <- cubist(x=as.matrix(italy_train[, c('Population', 'GDP', 'co2', 'energy_consumption')]),
                         y=italy_train$gpw, committees=5)

italy_cubist_fitted <- predict(italy_cubist, 
                                 newdata=as.data.frame(italy_train[, c('Population', 'GDP', 'co2', 'energy_consumption')]),
                                 neighbors=1)
mape(italy_cubist_fitted, italy_train$gpw)
rmse(italy_cubist_fitted, italy_train$gpw)
mae(italy_cubist_fitted, italy_train$gpw)

italy_cubist_forecast <- predict(italy_cubist,
                                   newdata=as.data.frame(italy_test[, c('Population', 'GDP', 'co2', 'energy_consumption')]),
                                   neighbors=1)
mape(italy_cubist_forecast, italy_test$gpw)
rmse(italy_cubist_forecast, italy_test$gpw)
mae(italy_cubist_forecast, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main='italy gpw Recycling - Cubist', col='black')
lines(y=italy_cubist_fitted, x=italy_train$Year, col='red', type='b')
lines(y=italy_cubist_forecast, x=italy_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
italy_bayes_ridge <- bayesglm(gpw~Population+GDP+co2+energy_consumption, data=italy_train,
                                family=gaussian)

italy_bayes_fitted <- predict(italy_bayes_ridge, newdata=italy_train)
mape(italy_bayes_fitted, italy_train$gpw)
rmse(italy_bayes_fitted, italy_train$gpw)
mae(italy_bayes_fitted, italy_train$gpw)

italy_bayes_forecast <- predict(italy_bayes_ridge, newdata=italy_test)
mape(italy_bayes_forecast, italy_test$gpw)
rmse(italy_bayes_forecast, italy_test$gpw)
mae(italy_bayes_forecast, italy_test$gpw)

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main='italy gpw Recycling - Bayesian Ridge', col='black')
lines(y=italy_bayes_fitted, x=italy_train$Year, col='red', type='b')
lines(y=italy_bayes_forecast, x=italy_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)



#luxembourg----
#train/test split (80/20)
luxembourg_train <- luxembourg[1:21, ]
luxembourg_test <- luxembourg[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(luxembourg_train)


#xarima
luxembourg_xarima <- auto.arima(y=luxembourg_train$gpw, max.p=5, max.q=5,
                                xreg=as.matrix(luxembourg_train[, c('Population', 'GDP', 'exports', 'energy_consumption')]))
mape(luxembourg_xarima$fitted, luxembourg_train$gpw)
rmse(luxembourg_xarima$fitted, luxembourg_train$gpw)
mae(luxembourg_xarima$fitted, luxembourg_train$gpw)

luxembourg_xarima_forecast <- forecast(luxembourg_xarima, h=nrow(luxembourg_test),
                                       xreg=as.matrix(luxembourg_test[, c('Population', 'GDP', 'exports', 'energy_consumption')]))

mape(luxembourg_xarima_forecast$mean, luxembourg_test$gpw)
rmse(luxembourg_xarima_forecast$mean, luxembourg_test$gpw)
mae(luxembourg_xarima_forecast$mean, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main= 'luxembourg gpw recycling rate - XARIMA', col='black')
lines(y=luxembourg_xarima$fitted, x=luxembourg_train$Year, col='red', type='b')
lines(y=luxembourg_xarima_forecast$mean, x=luxembourg_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~Population+GDP+exports+energy_consumption, 
                           data=luxembourg_train)
luxembourg_bsts <- bsts(gpw~Population+GDP+exports+energy_consumption,
                        state.specification=ss, data=luxembourg_train, niter=1000)

luxembourg_bsts_fitted <- predict(luxembourg_bsts, newdata=luxembourg_train,
                                  horizon=nrow(luxembourg_train), burn=100)
mape(luxembourg_bsts_fitted$mean, luxembourg_train$gpw)
rmse(luxembourg_bsts_fitted$mean, luxembourg_train$gpw)
mae(luxembourg_bsts_fitted$mean, luxembourg_train$gpw)

luxembourg_bsts_forecast <- predict(luxembourg_bsts, newdata=luxembourg_test, burn=100)
mape(luxembourg_bsts_forecast$mean, luxembourg_test$gpw)
rmse(luxembourg_bsts_forecast$mean, luxembourg_test$gpw)
mae(luxembourg_bsts_forecast$mean, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main='luxembourg gpw Recycling - BSTS', col='black')
lines(y=luxembourg_bsts_fitted$mean, x=luxembourg_train$Year, col='red', type='b')
lines(y=luxembourg_bsts_forecast$mean, x=luxembourg_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(luxembourg_train, c('Population', 'GDP', 'exports', 'energy_consumption'))
proc.time() - start

luxembourg_svm <- svm(gpw~Population+GDP+exports+energy_consumption, data=luxembourg_train, kernel='radial',
                      cost=3.5, gamma=0.05, epsilon=0.001)
mape(luxembourg_svm$fitted, luxembourg_train$gpw)
rmse(luxembourg_svm$fitted, luxembourg_train$gpw)
mae(luxembourg_svm$fitted, luxembourg_train$gpw)

luxembourg_svm_forecast <- predict(luxembourg_svm, newdata=luxembourg_test)
mape(luxembourg_svm_forecast, luxembourg_test$gpw)
rmse(luxembourg_svm_forecast, luxembourg_test$gpw)
mae(luxembourg_svm_forecast, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main='luxembourg gpw Recycling - SVM', col='black')
lines(y=luxembourg_svm$fitted, x=luxembourg_train$Year, col='red', type='b')
lines(y=luxembourg_svm_forecast, x=luxembourg_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(luxembourg_train, c('Population', 'GDP', 'exports', 'energy_consumption'))
proc.time()-start

set.seed(123)
luxembourg_rf <- randomForest(gpw~Population+GDP+exports+energy_consumption, data=luxembourg_train, maxnodes=8, ntree=50)

mape(luxembourg_rf$predicted, luxembourg_train$gpw)
rmse(luxembourg_rf$predicted, luxembourg_train$gpw)
mae(luxembourg_rf$predicted, luxembourg_train$gpw)

luxembourg_rf_forecast <- predict(luxembourg_rf, newdata=luxembourg_test)
mape(luxembourg_rf_forecast, luxembourg_test$gpw)
rmse(luxembourg_rf_forecast, luxembourg_test$gpw)
mae(luxembourg_rf_forecast, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main='luxembourg gpw Recycling - Radom Forest')
lines(y=luxembourg_rf$predicted, x=luxembourg_train$Year, type='b', col='red')
lines(y=luxembourg_rf_forecast, x=luxembourg_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(luxembourg_train, c('Population', 'GDP', 'exports', 'energy_consumption'))
proc.time()-start

luxembourg_X_train <- as.matrix(luxembourg_train[ , c('Population', 'GDP', 'exports', 'energy_consumption')])
luxembourg_y_train <- luxembourg_train$gpw
luxembourg_X_test  <- as.matrix(luxembourg_test[ , c('Population', 'GDP', 'exports', 'energy_consumption')])
luxembourg_y_test  <- luxembourg_test$gpw
luxembourg_dtrain <- xgb.DMatrix(data=luxembourg_X_train, label=luxembourg_y_train)
luxembourg_dtest  <- xgb.DMatrix(data=luxembourg_X_test)

set.seed(123)
luxembourg_xgb <- xgboost(data=luxembourg_dtrain, nrounds=100, max_depth=3,
                          eta=0.3, lambda=0.5, verbose=0)

luxembourg_xgb_fitted <- predict(luxembourg_xgb, luxembourg_dtrain)
mape(luxembourg_xgb_fitted, luxembourg_train$gpw)
rmse(luxembourg_xgb_fitted, luxembourg_train$gpw)
mae(luxembourg_xgb_fitted, luxembourg_train$gpw)

luxembourg_xgb_forecast <- predict(luxembourg_xgb, luxembourg_dtest)
mape(luxembourg_xgb_forecast, luxembourg_test$gpw)
rmse(luxembourg_xgb_forecast, luxembourg_test$gpw)
mae(luxembourg_xgb_forecast, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='gpw', main='luxembourg gpw - XGBoost')
lines(y=luxembourg_xgb_fitted, x=luxembourg_train$Year, type='b', col='red')
lines(y=luxembourg_xgb_forecast, x=luxembourg_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(luxembourg_train, c('Population', 'GDP', 'exports', 'energy_consumption'))
proc.time()-start

set.seed(123)
luxembourg_nnet <- nnetar(luxembourg_train$gpw, p=1, size=4, decay=0, 
                       repeats=10, xreg=as.matrix(luxembourg_train[, c('Population', 'GDP', 'exports', 'energy_consumption')]))

mape(as.numeric(fitted(luxembourg_nnet)[2:21]), luxembourg_train$gpw[2:21])
rmse(as.numeric(fitted(luxembourg_nnet)[2:21]), luxembourg_train$gpw[2:21])
mae(as.numeric(fitted(luxembourg_nnet)[2:21]), luxembourg_train$gpw[2:21])

luxembourg_nnet_forecast <- as.numeric(forecast(luxembourg_nnet, h=3, 
                                             xreg=as.matrix(luxembourg_test[, c('Population', 'GDP', 'exports', 'energy_consumption')]))$mean)

mape(luxembourg_nnet_forecast, luxembourg_test$gpw)
rmse(luxembourg_nnet_forecast, luxembourg_test$gpw)
mae(luxembourg_nnet_forecast, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main='luxembourg gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(luxembourg_nnet)), x=luxembourg_train$Year, type='b', col='red')
lines(y=luxembourg_nnet_forecast, x=luxembourg_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(luxembourg_train, c('Population', 'GDP', 'exports', 'energy_consumption'))
proc.time() - start

luxembourg_X_train <- as.matrix(luxembourg_train[, c('Population', 'GDP', 'exports', 'energy_consumption')])
luxembourg_X_test <- as.matrix(luxembourg_test[, c('Population', 'GDP', 'exports', 'energy_consumption')])

luxembourg_enet <- glmnet(luxembourg_X_train, luxembourg_train$gpw, alpha=0.5, lambda=0.5)

luxembourg_enet_fitted <- as.numeric(predict(luxembourg_enet, newx=luxembourg_X_train, s=0.5))
mape(luxembourg_enet_fitted, luxembourg_train$gpw)
rmse(luxembourg_enet_fitted, luxembourg_train$gpw)
mae(luxembourg_enet_fitted, luxembourg_train$gpw)

luxembourg_enet_forecast <- as.numeric(predict(luxembourg_enet, newx=luxembourg_X_test, s=0.5))
mape(luxembourg_enet_forecast, luxembourg_test$gpw)
rmse(luxembourg_enet_forecast, luxembourg_test$gpw)
mae(luxembourg_enet_forecast, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main='luxembourg gpw Recycling - Elastic Net', col='black')
lines(y=luxembourg_enet_fitted, x=luxembourg_train$Year, col='red', type='b')
lines(y=luxembourg_enet_forecast, x=luxembourg_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(luxembourg_train, c('Population', 'GDP', 'exports', 'energy_consumption'))
proc.time() - start

luxembourg_cubist <- cubist(x=as.matrix(luxembourg_train[, c('Population', 'GDP', 'exports', 'energy_consumption')]),
                       y=luxembourg_train$gpw, committees=1)

luxembourg_cubist_fitted <- predict(luxembourg_cubist, 
                               newdata=as.data.frame(luxembourg_train[, c('Population', 'GDP', 'exports', 'energy_consumption')]),
                               neighbors=1)
mape(luxembourg_cubist_fitted, luxembourg_train$gpw)
rmse(luxembourg_cubist_fitted, luxembourg_train$gpw)
mae(luxembourg_cubist_fitted, luxembourg_train$gpw)

luxembourg_cubist_forecast <- predict(luxembourg_cubist,
                                 newdata=as.data.frame(luxembourg_test[, c('Population', 'GDP', 'exports', 'energy_consumption')]),
                                 neighbors=1)
mape(luxembourg_cubist_forecast, luxembourg_test$gpw)
rmse(luxembourg_cubist_forecast, luxembourg_test$gpw)
mae(luxembourg_cubist_forecast, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main='luxembourg gpw Recycling - Cubist', col='black')
lines(y=luxembourg_cubist_fitted, x=luxembourg_train$Year, col='red', type='b')
lines(y=luxembourg_cubist_forecast, x=luxembourg_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
luxembourg_bayes_ridge <- bayesglm(gpw~Population+GDP+exports+energy_consumption, data=luxembourg_train,
                              family=gaussian)

luxembourg_bayes_fitted <- predict(luxembourg_bayes_ridge, newdata=luxembourg_train)
mape(luxembourg_bayes_fitted, luxembourg_train$gpw)
rmse(luxembourg_bayes_fitted, luxembourg_train$gpw)
mae(luxembourg_bayes_fitted, luxembourg_train$gpw)

luxembourg_bayes_forecast <- predict(luxembourg_bayes_ridge, newdata=luxembourg_test)
mape(luxembourg_bayes_forecast, luxembourg_test$gpw)
rmse(luxembourg_bayes_forecast, luxembourg_test$gpw)
mae(luxembourg_bayes_forecast, luxembourg_test$gpw)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main='luxembourg gpw Recycling - Bayesian Ridge', col='black')
lines(y=luxembourg_bayes_fitted, x=luxembourg_train$Year, col='red', type='b')
lines(y=luxembourg_bayes_forecast, x=luxembourg_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#netherlands----
#train/test split (80/20)
netherlands_train <- netherlands[1:21, ]
netherlands_test <- netherlands[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(netherlands_train)


#xarima
netherlands_xarima <- auto.arima(y=netherlands_train$gpw, max.p=5, max.q=5,
                                 xreg=as.matrix(netherlands_train[, c('GDP')]))
mape(netherlands_xarima$fitted, netherlands_train$gpw)
rmse(netherlands_xarima$fitted, netherlands_train$gpw)
mae(netherlands_xarima$fitted, netherlands_train$gpw)

netherlands_xarima_forecast <- forecast(netherlands_xarima, h=nrow(netherlands_test),
                                        xreg=as.matrix(netherlands_test[, c('GDP')]))

mape(netherlands_xarima_forecast$mean, netherlands_test$gpw)
rmse(netherlands_xarima_forecast$mean, netherlands_test$gpw)
mae(netherlands_xarima_forecast$mean, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main= 'netherlands gpw recycling rate - XARIMA', col='black')
lines(y=netherlands_xarima$fitted, x=netherlands_train$Year, col='red', type='b')
lines(y=netherlands_xarima_forecast$mean, x=netherlands_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~GDP, 
                           data=netherlands_train)
netherlands_bsts <- bsts(gpw~GDP,
                         state.specification=ss, data=netherlands_train, niter=1000)

netherlands_bsts_fitted <- predict(netherlands_bsts, newdata=netherlands_train,
                                   horizon=nrow(netherlands_train), burn=100)
mape(netherlands_bsts_fitted$mean, netherlands_train$gpw)
rmse(netherlands_bsts_fitted$mean, netherlands_train$gpw)
mae(netherlands_bsts_fitted$mean, netherlands_train$gpw)

netherlands_bsts_forecast <- predict(netherlands_bsts, newdata=netherlands_test, burn=100)
mape(netherlands_bsts_forecast$mean, netherlands_test$gpw)
rmse(netherlands_bsts_forecast$mean, netherlands_test$gpw)
mae(netherlands_bsts_forecast$mean, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main='netherlands gpw Recycling - BSTS', col='black')
lines(y=netherlands_bsts_fitted$mean, x=netherlands_train$Year, col='red', type='b')
lines(y=netherlands_bsts_forecast$mean, x=netherlands_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(netherlands_train, c('GDP'))
proc.time() - start

netherlands_svm <- svm(gpw~GDP, data=netherlands_train, kernel='radial',
                       cost=1, gamma=0.05, epsilon=0.041)
mape(netherlands_svm$fitted, netherlands_train$gpw)
rmse(netherlands_svm$fitted, netherlands_train$gpw)
mae(netherlands_svm$fitted, netherlands_train$gpw)

netherlands_svm_forecast <- predict(netherlands_svm, newdata=netherlands_test)
mape(netherlands_svm_forecast, netherlands_test$gpw)
rmse(netherlands_svm_forecast, netherlands_test$gpw)
mae(netherlands_svm_forecast, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main='netherlands gpw Recycling - SVM', col='black')
lines(y=netherlands_svm$fitted, x=netherlands_train$Year, col='red', type='b')
lines(y=netherlands_svm_forecast, x=netherlands_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(netherlands_train, c('GDP'))
proc.time()-start

set.seed(123)
netherlands_rf <- randomForest(gpw~GDP, data=netherlands_train, maxnodes=1, ntree=100)

mape(netherlands_rf$predicted, netherlands_train$gpw)
rmse(netherlands_rf$predicted, netherlands_train$gpw)
mae(netherlands_rf$predicted, netherlands_train$gpw)

netherlands_rf_forecast <- predict(netherlands_rf, newdata=netherlands_test)
mape(netherlands_rf_forecast, netherlands_test$gpw)
rmse(netherlands_rf_forecast, netherlands_test$gpw)
mae(netherlands_rf_forecast, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main='netherlands gpw Recycling - Radom Forest')
lines(y=netherlands_rf$predicted, x=netherlands_train$Year, type='b', col='red')
lines(y=netherlands_rf_forecast, x=netherlands_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(netherlands_train, c('GDP'))
proc.time()-start

netherlands_X_train <- as.matrix(netherlands_train[ , c('GDP')])
netherlands_y_train <- netherlands_train$gpw
netherlands_X_test  <- as.matrix(netherlands_test[ , c('GDP')])
netherlands_y_test  <- netherlands_test$gpw
netherlands_dtrain <- xgb.DMatrix(data=netherlands_X_train, label=netherlands_y_train)
netherlands_dtest  <- xgb.DMatrix(data=netherlands_X_test)

set.seed(123)
netherlands_xgb <- xgboost(data=netherlands_dtrain, nrounds=100, max_depth=3,
                           eta=0.05, lambda=5, verbose=0)

netherlands_xgb_fitted <- predict(netherlands_xgb, netherlands_dtrain)
mape(netherlands_xgb_fitted, netherlands_train$gpw)
rmse(netherlands_xgb_fitted, netherlands_train$gpw)
mae(netherlands_xgb_fitted, netherlands_train$gpw)

netherlands_xgb_forecast <- predict(netherlands_xgb, netherlands_dtest)
mape(netherlands_xgb_forecast, netherlands_test$gpw)
rmse(netherlands_xgb_forecast, netherlands_test$gpw)
mae(netherlands_xgb_forecast, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='gpw', main='netherlands gpw - XGBoost')
lines(y=netherlands_xgb_fitted, x=netherlands_train$Year, type='b', col='red')
lines(y=netherlands_xgb_forecast, x=netherlands_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(netherlands_train, c('GDP'))
proc.time()-start

set.seed(123)
netherlands_nnet <- nnetar(netherlands_train$gpw, p=1, size=1, decay=0.1, 
                       repeats=20, xreg=as.matrix(netherlands_train[, c('GDP')]))

mape(as.numeric(fitted(netherlands_nnet)[2:21]), netherlands_train$gpw[2:21])
rmse(as.numeric(fitted(netherlands_nnet)[2:21]), netherlands_train$gpw[2:21])
mae(as.numeric(fitted(netherlands_nnet)[2:21]), netherlands_train$gpw[2:21])

netherlands_nnet_forecast <- as.numeric(forecast(netherlands_nnet, h=3, 
                                             xreg=as.matrix(netherlands_test[, c('GDP')]))$mean)

mape(netherlands_nnet_forecast, netherlands_test$gpw)
rmse(netherlands_nnet_forecast, netherlands_test$gpw)
mae(netherlands_nnet_forecast, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main='netherlands gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(netherlands_nnet)), x=netherlands_train$Year, type='b', col='red')
lines(y=netherlands_nnet_forecast, x=netherlands_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(netherlands_train, c('GDP'))
proc.time() - start

netherlands_X_train <- as.matrix(netherlands_train[, c('GDP')])
netherlands_X_test <- as.matrix(netherlands_test[, c('GDP')])

netherlands_enet <- glmnet(netherlands_X_train, netherlands_train$gpw, alpha=0.5, lambda=0.5)

netherlands_enet_fitted <- as.numeric(predict(netherlands_enet, newx=netherlands_X_train, s=0.5))
mape(netherlands_enet_fitted, netherlands_train$gpw)
rmse(netherlands_enet_fitted, netherlands_train$gpw)
mae(netherlands_enet_fitted, netherlands_train$gpw)

netherlands_enet_forecast <- as.numeric(predict(netherlands_enet, newx=netherlands_X_test, s=0.5))
mape(netherlands_enet_forecast, netherlands_test$gpw)
rmse(netherlands_enet_forecast, netherlands_test$gpw)
mae(netherlands_enet_forecast, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main='netherlands gpw Recycling - Elastic Net', col='black')
lines(y=netherlands_enet_fitted, x=netherlands_train$Year, col='red', type='b')
lines(y=netherlands_enet_forecast, x=netherlands_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(netherlands_train, c('GDP'))
proc.time() - start

netherlands_cubist <- cubist(x=as.matrix(netherlands_train[, c('GDP')]),
                            y=netherlands_train$gpw, committees=10)

netherlands_cubist_fitted <- predict(netherlands_cubist, 
                                    newdata=as.data.frame(netherlands_train[, c('GDP')]),
                                    neighbors=0)
mape(netherlands_cubist_fitted, netherlands_train$gpw)
rmse(netherlands_cubist_fitted, netherlands_train$gpw)
mae(netherlands_cubist_fitted, netherlands_train$gpw)

netherlands_cubist_forecast <- predict(netherlands_cubist,
                                      newdata=as.data.frame(netherlands_test[, c('GDP')]),
                                      neighbors=0)
mape(netherlands_cubist_forecast, netherlands_test$gpw)
rmse(netherlands_cubist_forecast, netherlands_test$gpw)
mae(netherlands_cubist_forecast, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main='netherlands gpw Recycling - Cubist', col='black')
lines(y=netherlands_cubist_fitted, x=netherlands_train$Year, col='red', type='b')
lines(y=netherlands_cubist_forecast, x=netherlands_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
netherlands_bayes_ridge <- bayesglm(gpw~GDP, data=netherlands_train,
                                   family=gaussian)

netherlands_bayes_fitted <- predict(netherlands_bayes_ridge, newdata=netherlands_train)
mape(netherlands_bayes_fitted, netherlands_train$gpw)
rmse(netherlands_bayes_fitted, netherlands_train$gpw)
mae(netherlands_bayes_fitted, netherlands_train$gpw)

netherlands_bayes_forecast <- predict(netherlands_bayes_ridge, newdata=netherlands_test)
mape(netherlands_bayes_forecast, netherlands_test$gpw)
rmse(netherlands_bayes_forecast, netherlands_test$gpw)
mae(netherlands_bayes_forecast, netherlands_test$gpw)

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main='netherlands gpw Recycling - Bayesian Ridge', col='black')
lines(y=netherlands_bayes_fitted, x=netherlands_train$Year, col='red', type='b')
lines(y=netherlands_bayes_forecast, x=netherlands_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#portugal----
#train/test split (80/20)
portugal_train <- portugal[2:21, ] #this is 2 as portugal has no 1997 figure
portugal_train$gpw <- as.double(portugal_train$gpw)
portugal_test <- portugal[22:26, ]
portugal_test$gpw <- as.double(portugal_test$gpw)

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(portugal_train)


#xarima
portugal_xarima <- auto.arima(y=portugal_train$gpw, max.p=5, max.q=5,
                              xreg=as.matrix(portugal_train[, c('co2')]))
mape(portugal_xarima$fitted, portugal_train$gpw)
rmse(portugal_xarima$fitted, portugal_train$gpw)
mae(portugal_xarima$fitted, portugal_train$gpw)

portugal_xarima_forecast <- forecast(portugal_xarima, h=nrow(portugal_test),
                                     xreg=as.matrix(portugal_test[, c('co2')]))

mape(portugal_xarima_forecast$mean, portugal_test$gpw)
rmse(portugal_xarima_forecast$mean, portugal_test$gpw)
mae(portugal_xarima_forecast$mean, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main= 'portugal gpw recycling rate - XARIMA', col='black')
lines(y=portugal_xarima$fitted, x=portugal_train$Year, col='red', type='b')
lines(y=portugal_xarima_forecast$mean, x=portugal_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~co2, 
                           data=portugal_train)
portugal_bsts <- bsts(gpw~co2,
                      state.specification=ss, data=portugal_train, niter=1000)

portugal_bsts_fitted <- predict(portugal_bsts, newdata=portugal_train,
                                horizon=nrow(portugal_train), burn=100)
mape(portugal_bsts_fitted$mean, portugal_train$gpw)
rmse(portugal_bsts_fitted$mean, portugal_train$gpw)
mae(portugal_bsts_fitted$mean, portugal_train$gpw)

portugal_bsts_forecast <- predict(portugal_bsts, newdata=portugal_test, burn=100)
mape(portugal_bsts_forecast$mean, portugal_test$gpw)
rmse(portugal_bsts_forecast$mean, portugal_test$gpw)
mae(portugal_bsts_forecast$mean, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main='portugal gpw Recycling - BSTS', col='black')
lines(y=portugal_bsts_fitted$mean, x=portugal_train$Year, col='red', type='b')
lines(y=portugal_bsts_forecast$mean, x=portugal_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(portugal_train, c('co2'))
proc.time() - start

portugal_svm <- svm(gpw~co2, data=portugal_train, kernel='radial',
                    cost=0.3, gamma=0.3, epsilon=0.071)
mape(portugal_svm$fitted, portugal_train$gpw)
rmse(portugal_svm$fitted, portugal_train$gpw)
mae(portugal_svm$fitted, portugal_train$gpw)

portugal_svm_forecast <- predict(portugal_svm, newdata=portugal_test)
mape(portugal_svm_forecast, portugal_test$gpw)
rmse(portugal_svm_forecast, portugal_test$gpw)
mae(portugal_svm_forecast, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main='portugal gpw Recycling - SVM', col='black')
lines(y=portugal_svm$fitted, x=portugal_train$Year, col='red', type='b')
lines(y=portugal_svm_forecast, x=portugal_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(portugal_train, c('co2'))
proc.time()-start

set.seed(123)
portugal_rf <- randomForest(gpw~co2, data=portugal_train, maxnodes=6, ntree=50)

mape(portugal_rf$predicted, portugal_train$gpw)
rmse(portugal_rf$predicted, portugal_train$gpw)
mae(portugal_rf$predicted, portugal_train$gpw)

portugal_rf_forecast <- predict(portugal_rf, newdata=portugal_test)
mape(portugal_rf_forecast, portugal_test$gpw)
rmse(portugal_rf_forecast, portugal_test$gpw)
mae(portugal_rf_forecast, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main='portugal gpw Recycling - Radom Forest')
lines(y=portugal_rf$predicted, x=portugal_train$Year, type='b', col='red')
lines(y=portugal_rf_forecast, x=portugal_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(portugal_train, c('co2'))
proc.time()-start

portugal_X_train <- as.matrix(portugal_train[ , c('co2')])
portugal_y_train <- portugal_train$gpw
portugal_X_test  <- as.matrix(portugal_test[ , c('co2')])
portugal_y_test  <- portugal_test$gpw
portugal_dtrain <- xgb.DMatrix(data=portugal_X_train, label=portugal_y_train)
portugal_dtest  <- xgb.DMatrix(data=portugal_X_test)

set.seed(123)
portugal_xgb <- xgboost(data=portugal_dtrain, nrounds=100, max_depth=2,
                        eta=0.05, lambda=2, verbose=0)

portugal_xgb_fitted <- predict(portugal_xgb, portugal_dtrain)
mape(portugal_xgb_fitted, portugal_train$gpw)
rmse(portugal_xgb_fitted, portugal_train$gpw)
mae(portugal_xgb_fitted, portugal_train$gpw)

portugal_xgb_forecast <- predict(portugal_xgb, portugal_dtest)
mape(portugal_xgb_forecast, portugal_test$gpw)
rmse(portugal_xgb_forecast, portugal_test$gpw)
mae(portugal_xgb_forecast, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='gpw', main='portugal gpw - XGBoost')
lines(y=portugal_xgb_fitted, x=portugal_train$Year, type='b', col='red')
lines(y=portugal_xgb_forecast, x=portugal_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(portugal_train, c('co2'))
proc.time()-start

set.seed(123)
portugal_nnet <- nnetar(portugal_train$gpw, p=1, size=1, decay=0.1, 
                       repeats=10, xreg=as.matrix(portugal_train[, c('co2')]))

mape(as.numeric(fitted(portugal_nnet)[2:20]), portugal_train$gpw[2:20])
rmse(as.numeric(fitted(portugal_nnet)[2:20]), portugal_train$gpw[2:20])
mae(as.numeric(fitted(portugal_nnet)[2:20]), portugal_train$gpw[2:20])

portugal_nnet_forecast <- as.numeric(forecast(portugal_nnet, h=3, 
                                             xreg=as.matrix(portugal_test[, c('co2')]))$mean)

mape(portugal_nnet_forecast, portugal_test$gpw)
rmse(portugal_nnet_forecast, portugal_test$gpw)
mae(portugal_nnet_forecast, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main='portugal gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(portugal_nnet)), x=portugal_train$Year, type='b', col='red')
lines(y=portugal_nnet_forecast, x=portugal_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(portugal_train, c('co2'))
proc.time() - start

portugal_X_train <- as.matrix(portugal_train[, c('co2')])
portugal_X_test <- as.matrix(portugal_test[, c('co2')])

portugal_enet <- glmnet(portugal_X_train, portugal_train$gpw, alpha=0.5, lambda=0.5)

portugal_enet_fitted <- as.numeric(predict(portugal_enet, newx=portugal_X_train, s=0.5))
mape(portugal_enet_fitted, portugal_train$gpw)
rmse(portugal_enet_fitted, portugal_train$gpw)
mae(portugal_enet_fitted, portugal_train$gpw)

portugal_enet_forecast <- as.numeric(predict(portugal_enet, newx=portugal_X_test, s=0.5))
mape(portugal_enet_forecast, portugal_test$gpw)
rmse(portugal_enet_forecast, portugal_test$gpw)
mae(portugal_enet_forecast, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main='portugal gpw Recycling - Elastic Net', col='black')
lines(y=portugal_enet_fitted, x=portugal_train$Year, col='red', type='b')
lines(y=portugal_enet_forecast, x=portugal_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(portugal_train, c('co2'))
proc.time() - start

portugal_cubist <- cubist(x=as.matrix(portugal_train[, c('co2')]),
                             y=portugal_train$gpw, committees=10)

portugal_cubist_fitted <- predict(portugal_cubist, 
                                     newdata=as.data.frame(portugal_train[, c('co2')]),
                                     neighbors=9)
mape(portugal_cubist_fitted, portugal_train$gpw)
rmse(portugal_cubist_fitted, portugal_train$gpw)
mae(portugal_cubist_fitted, portugal_train$gpw)

portugal_cubist_forecast <- predict(portugal_cubist,
                                       newdata=as.data.frame(portugal_test[, c('co2')]),
                                       neighbors=9)
mape(portugal_cubist_forecast, portugal_test$gpw)
rmse(portugal_cubist_forecast, portugal_test$gpw)
mae(portugal_cubist_forecast, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main='portugal gpw Recycling - Cubist', col='black')
lines(y=portugal_cubist_fitted, x=portugal_train$Year, col='red', type='b')
lines(y=portugal_cubist_forecast, x=portugal_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
portugal_bayes_ridge <- bayesglm(gpw~co2, data=portugal_train,
                                    family=gaussian)

portugal_bayes_fitted <- predict(portugal_bayes_ridge, newdata=portugal_train)
mape(portugal_bayes_fitted, portugal_train$gpw)
rmse(portugal_bayes_fitted, portugal_train$gpw)
mae(portugal_bayes_fitted, portugal_train$gpw)

portugal_bayes_forecast <- predict(portugal_bayes_ridge, newdata=portugal_test)
mape(portugal_bayes_forecast, portugal_test$gpw)
rmse(portugal_bayes_forecast, portugal_test$gpw)
mae(portugal_bayes_forecast, portugal_test$gpw)

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main='portugal gpw Recycling - Bayesian Ridge', col='black')
lines(y=portugal_bayes_fitted, x=portugal_train$Year, col='red', type='b')
lines(y=portugal_bayes_forecast, x=portugal_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#spain----
#train/test split (80/20)
spain_train <- spain[1:21, ]
spain_test <- spain[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(spain_train)


#xarima
spain_xarima <- auto.arima(y=spain_train$gpw, max.p=5, max.q=5,
                           xreg=as.matrix(spain_train[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')]))
mape(spain_xarima$fitted, spain_train$gpw)
rmse(spain_xarima$fitted, spain_train$gpw)
mae(spain_xarima$fitted, spain_train$gpw)

spain_xarima_forecast <- forecast(spain_xarima, h=nrow(spain_test),
                                  xreg=as.matrix(spain_test[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')]))

mape(spain_xarima_forecast$mean, spain_test$gpw)
rmse(spain_xarima_forecast$mean, spain_test$gpw)
mae(spain_xarima_forecast$mean, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main= 'spain gpw recycling rate - XARIMA', col='black')
lines(y=spain_xarima$fitted, x=spain_train$Year, col='red', type='b')
lines(y=spain_xarima_forecast$mean, x=spain_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~Population+GDP+co2+material_footprint+exports+energy_consumption, 
                           data=spain_train)
spain_bsts <- bsts(gpw~Population+GDP+co2+material_footprint+exports+energy_consumption,
                   state.specification=ss, data=spain_train, niter=1000)

spain_bsts_fitted <- predict(spain_bsts, newdata=spain_train,
                             horizon=nrow(spain_train), burn=100)
mape(spain_bsts_fitted$mean, spain_train$gpw)
rmse(spain_bsts_fitted$mean, spain_train$gpw)
mae(spain_bsts_fitted$mean, spain_train$gpw)

spain_bsts_forecast <- predict(spain_bsts, newdata=spain_test, burn=100)
mape(spain_bsts_forecast$mean, spain_test$gpw)
rmse(spain_bsts_forecast$mean, spain_test$gpw)
mae(spain_bsts_forecast$mean, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main='spain gpw Recycling - BSTS', col='black')
lines(y=spain_bsts_fitted$mean, x=spain_train$Year, col='red', type='b')
lines(y=spain_bsts_forecast$mean, x=spain_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(spain_train, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption'))
proc.time() - start

spain_svm <- svm(gpw~Population+GDP+co2+material_footprint+exports+energy_consumption, data=spain_train, kernel='radial',
                 cost=3.7, gamma=0.05, epsilon=0.001)
mape(spain_svm$fitted, spain_train$gpw)
rmse(spain_svm$fitted, spain_train$gpw)
mae(spain_svm$fitted, spain_train$gpw)

spain_svm_forecast <- predict(spain_svm, newdata=spain_test)
mape(spain_svm_forecast, spain_test$gpw)
rmse(spain_svm_forecast, spain_test$gpw)
mae(spain_svm_forecast, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main='spain gpw Recycling - SVM', col='black')
lines(y=spain_svm$fitted, x=spain_train$Year, col='red', type='b')
lines(y=spain_svm_forecast, x=spain_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(spain_train, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption'))
proc.time()-start

set.seed(123)
spain_rf <- randomForest(gpw~Population+GDP+co2+material_footprint+exports+energy_consumption, data=spain_train, maxnodes=6, ntree=100)

mape(spain_rf$predicted, spain_train$gpw)
rmse(spain_rf$predicted, spain_train$gpw)
mae(spain_rf$predicted, spain_train$gpw)

spain_rf_forecast <- predict(spain_rf, newdata=spain_test)
mape(spain_rf_forecast, spain_test$gpw)
rmse(spain_rf_forecast, spain_test$gpw)
mae(spain_rf_forecast, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main='spain gpw Recycling - Radom Forest')
lines(y=spain_rf$predicted, x=spain_train$Year, type='b', col='red')
lines(y=spain_rf_forecast, x=spain_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(spain_train, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption'))
proc.time()-start

spain_X_train <- as.matrix(spain_train[ , c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')])
spain_y_train <- spain_train$gpw
spain_X_test  <- as.matrix(spain_test[ , c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')])
spain_y_test  <- spain_test$gpw
spain_dtrain <- xgb.DMatrix(data=spain_X_train, label=spain_y_train)
spain_dtest  <- xgb.DMatrix(data=spain_X_test)

set.seed(123)
spain_xgb <- xgboost(data=spain_dtrain, nrounds=100, max_depth=3,
                     eta=0.1, lambda=0.5, verbose=0)

spain_xgb_fitted <- predict(spain_xgb, spain_dtrain)
mape(spain_xgb_fitted, spain_train$gpw)
rmse(spain_xgb_fitted, spain_train$gpw)
mae(spain_xgb_fitted, spain_train$gpw)

spain_xgb_forecast <- predict(spain_xgb, spain_dtest)
mape(spain_xgb_forecast, spain_test$gpw)
rmse(spain_xgb_forecast, spain_test$gpw)
mae(spain_xgb_forecast, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='gpw', main='spain gpw - XGBoost')
lines(y=spain_xgb_fitted, x=spain_train$Year, type='b', col='red')
lines(y=spain_xgb_forecast, x=spain_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(spain_train, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption'))
proc.time()-start

set.seed(123)
spain_nnet <- nnetar(spain_train$gpw, p=1, size=5, decay=0.1, 
                       repeats=10, xreg=as.matrix(spain_train[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')]))

mape(as.numeric(fitted(spain_nnet)[2:21]), spain_train$gpw[2:21])
rmse(as.numeric(fitted(spain_nnet)[2:21]), spain_train$gpw[2:21])
mae(as.numeric(fitted(spain_nnet)[2:21]), spain_train$gpw[2:21])

spain_nnet_forecast <- as.numeric(forecast(spain_nnet, h=3, 
                                             xreg=as.matrix(spain_test[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')]))$mean)

mape(spain_nnet_forecast, spain_test$gpw)
rmse(spain_nnet_forecast, spain_test$gpw)
mae(spain_nnet_forecast, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main='spain gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(spain_nnet)), x=spain_train$Year, type='b', col='red')
lines(y=spain_nnet_forecast, x=spain_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#elastic net
start <- proc.time()
rolling_cv_enet(spain_train, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption'))
proc.time() - start

spain_X_train <- as.matrix(spain_train[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')])
spain_X_test <- as.matrix(spain_test[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')])

spain_enet <- glmnet(spain_X_train, spain_train$gpw, alpha=0.7, lambda=0.1)

spain_enet_fitted <- as.numeric(predict(spain_enet, newx=spain_X_train, s=0.1))
mape(spain_enet_fitted, spain_train$gpw)
rmse(spain_enet_fitted, spain_train$gpw)
mae(spain_enet_fitted, spain_train$gpw)

spain_enet_forecast <- as.numeric(predict(spain_enet, newx=spain_X_test, s=0.1))
mape(spain_enet_forecast, spain_test$gpw)
rmse(spain_enet_forecast, spain_test$gpw)
mae(spain_enet_forecast, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main='spain gpw Recycling - Elastic Net', col='black')
lines(y=spain_enet_fitted, x=spain_train$Year, col='red', type='b')
lines(y=spain_enet_forecast, x=spain_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(spain_train, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption'))
proc.time() - start

spain_cubist <- cubist(x=as.matrix(spain_train[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')]),
                          y=spain_train$gpw, committees=1)

spain_cubist_fitted <- predict(spain_cubist, 
                                  newdata=as.data.frame(spain_train[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')]),
                                  neighbors=1)
mape(spain_cubist_fitted, spain_train$gpw)
rmse(spain_cubist_fitted, spain_train$gpw)
mae(spain_cubist_fitted, spain_train$gpw)

spain_cubist_forecast <- predict(spain_cubist,
                                    newdata=as.data.frame(spain_test[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')]),
                                    neighbors=1)
mape(spain_cubist_forecast, spain_test$gpw)
rmse(spain_cubist_forecast, spain_test$gpw)
mae(spain_cubist_forecast, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main='spain gpw Recycling - Cubist', col='black')
lines(y=spain_cubist_fitted, x=spain_train$Year, col='red', type='b')
lines(y=spain_cubist_forecast, x=spain_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
spain_bayes_ridge <- bayesglm(gpw~Population+GDP+co2+material_footprint+exports+energy_consumption, data=spain_train,
                                 family=gaussian)

spain_bayes_fitted <- predict(spain_bayes_ridge, newdata=spain_train)
mape(spain_bayes_fitted, spain_train$gpw)
rmse(spain_bayes_fitted, spain_train$gpw)
mae(spain_bayes_fitted, spain_train$gpw)

spain_bayes_forecast <- predict(spain_bayes_ridge, newdata=spain_test)
mape(spain_bayes_forecast, spain_test$gpw)
rmse(spain_bayes_forecast, spain_test$gpw)
mae(spain_bayes_forecast, spain_test$gpw)

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main='spain gpw Recycling - Bayesian Ridge', col='black')
lines(y=spain_bayes_fitted, x=spain_train$Year, col='red', type='b')
lines(y=spain_bayes_forecast, x=spain_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)



#sweden----
#train/test split (80/20)
sweden_train <- sweden[1:21, ]
sweden_test <- sweden[22:26, ]

#call lasso function to determine covariates to move forward with.
lasso_covariate_selection(sweden_train)


#xarima
sweden_xarima <- auto.arima(y=sweden_train$gpw, max.p=5, max.q=5,
                            xreg=as.matrix(sweden_train[, c('GDP','co2', 'material_footprint', 'energy_consumption')]))
mape(sweden_xarima$fitted, sweden_train$gpw)
rmse(sweden_xarima$fitted, sweden_train$gpw)
mae(sweden_xarima$fitted, sweden_train$gpw)

sweden_xarima_forecast <- forecast(sweden_xarima, h=nrow(sweden_test),
                                   xreg=as.matrix(sweden_test[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')]))

mape(sweden_xarima_forecast$mean, sweden_test$gpw)
rmse(sweden_xarima_forecast$mean, sweden_test$gpw)
mae(sweden_xarima_forecast$mean, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main= 'sweden gpw recycling rate - XARIMA', col='black')
lines(y=sweden_xarima$fitted, x=sweden_train$Year, col='red', type='b')
lines(y=sweden_xarima_forecast$mean, x=sweden_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#bsts
set.seed(20229798)
ss <- AddDynamicRegression(list(), gpw~GDP+co2+material_footprint+energy_consumption, 
                           data=sweden_train)
sweden_bsts <- bsts(gpw~GDP+co2+material_footprint+energy_consumption,
                    state.specification=ss, data=sweden_train, niter=1000)

sweden_bsts_fitted <- predict(sweden_bsts, newdata=sweden_train,
                              horizon=nrow(sweden_train), burn=100)
mape(sweden_bsts_fitted$mean, sweden_train$gpw)
rmse(sweden_bsts_fitted$mean, sweden_train$gpw)
mae(sweden_bsts_fitted$mean, sweden_train$gpw)

sweden_bsts_forecast <- predict(sweden_bsts, newdata=sweden_test, burn=100)
mape(sweden_bsts_forecast$mean, sweden_test$gpw)
rmse(sweden_bsts_forecast$mean, sweden_test$gpw)
mae(sweden_bsts_forecast$mean, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main='sweden gpw Recycling - BSTS', col='black')
lines(y=sweden_bsts_fitted$mean, x=sweden_train$Year, col='red', type='b')
lines(y=sweden_bsts_forecast$mean, x=sweden_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#svm
start <- proc.time()
rolling_cv_svm(sweden_train, c('GDP', 'co2', 'material_footprint', 'energy_consumption'))
proc.time() - start

sweden_svm <- svm(gpw~GDP+co2+material_footprint+energy_consumption, data=sweden_train, kernel='radial',
                  cost=5, gamma=0.05, epsilon=0.001)
mape(sweden_svm$fitted, sweden_train$gpw)
rmse(sweden_svm$fitted, sweden_train$gpw)
mae(sweden_svm$fitted, sweden_train$gpw)

sweden_svm_forecast <- predict(sweden_svm, newdata=sweden_test)
mape(sweden_svm_forecast, sweden_test$gpw)
rmse(sweden_svm_forecast, sweden_test$gpw)
mae(sweden_svm_forecast, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main='sweden gpw Recycling - SVM', col='black')
lines(y=sweden_svm$fitted, x=sweden_train$Year, col='red', type='b')
lines(y=sweden_svm_forecast, x=sweden_test$Year, col='blue', type='b')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#rf
start <- proc.time()
rolling_cv_rf(sweden_train, c('GDP', 'co2', 'material_footprint', 'energy_consumption'))
proc.time()-start

set.seed(123)
sweden_rf <- randomForest(gpw~GDP+co2+material_footprint+energy_consumption, data=sweden_train, maxnodes=3, ntree=350)

mape(sweden_rf$predicted, sweden_train$gpw)
rmse(sweden_rf$predicted, sweden_train$gpw)
mae(sweden_rf$predicted, sweden_train$gpw)

sweden_rf_forecast <- predict(sweden_rf, newdata=sweden_test)
mape(sweden_rf_forecast, sweden_test$gpw)
rmse(sweden_rf_forecast, sweden_test$gpw)
mae(sweden_rf_forecast, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main='sweden gpw Recycling - Radom Forest')
lines(y=sweden_rf$predicted, x=sweden_train$Year, type='b', col='red')
lines(y=sweden_rf_forecast, x=sweden_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)


#XGBoost
start <- proc.time()
rolling_cv_xgb(sweden_train, c('GDP', 'co2', 'material_footprint', 'energy_consumption'))
proc.time()-start

sweden_X_train <- as.matrix(sweden_train[ , c('GDP', 'co2', 'material_footprint', 'energy_consumption')])
sweden_y_train <- sweden_train$gpw
sweden_X_test  <- as.matrix(sweden_test[ , c('GDP', 'co2', 'material_footprint', 'energy_consumption')])
sweden_y_test  <- sweden_test$gpw
sweden_dtrain <- xgb.DMatrix(data=sweden_X_train, label=sweden_y_train)
sweden_dtest  <- xgb.DMatrix(data=sweden_X_test)

set.seed(123)
sweden_xgb <- xgboost(data=sweden_dtrain, nrounds=100, max_depth=2,
                      eta=0.05, lambda=0.1, verbose=0)

sweden_xgb_fitted <- predict(sweden_xgb, sweden_dtrain)
mape(sweden_xgb_fitted, sweden_train$gpw)
rmse(sweden_xgb_fitted, sweden_train$gpw)
mae(sweden_xgb_fitted, sweden_train$gpw)

sweden_xgb_forecast <- predict(sweden_xgb, sweden_dtest)
mape(sweden_xgb_forecast, sweden_test$gpw)
rmse(sweden_xgb_forecast, sweden_test$gpw)
mae(sweden_xgb_forecast, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='gpw', main='sweden gpw - XGBoost')
lines(y=sweden_xgb_fitted, x=sweden_train$Year, type='b', col='red')
lines(y=sweden_xgb_forecast, x=sweden_test$Year, type='b', col='blue')
legend("topleft", legend = c("Actual", "Fitted", "Forecast"), 
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n",cex=0.75)

#nnet
start <- proc.time()
rolling_cv_nnet(sweden_train, c('GDP', 'co2', 'material_footprint', 'energy_consumption'))
proc.time()-start

set.seed(123)
sweden_nnet <- nnetar(sweden_train$gpw, p=1, size=2, decay=0, 
                       repeats=50, xreg=as.matrix(sweden_train[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')]))

mape(as.numeric(fitted(sweden_nnet)[2:21]), sweden_train$gpw[2:21])
rmse(as.numeric(fitted(sweden_nnet)[2:21]), sweden_train$gpw[2:21])
mae(as.numeric(fitted(sweden_nnet)[2:21]), sweden_train$gpw[2:21])

sweden_nnet_forecast <- as.numeric(forecast(sweden_nnet, h=3, 
                                             xreg=as.matrix(sweden_test[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')]))$mean)

mape(sweden_nnet_forecast, sweden_test$gpw)
rmse(sweden_nnet_forecast, sweden_test$gpw)
mae(sweden_nnet_forecast, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main='sweden gpw Recycling - Neural Network')
lines(y=as.numeric(fitted(sweden_nnet)), x=sweden_train$Year, type='b', col='red')
lines(y=sweden_nnet_forecast, x=sweden_test$Year, type='b', col='blue')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#elastic net
start <- proc.time()
rolling_cv_enet(sweden_train, c('GDP', 'co2', 'material_footprint', 'energy_consumption'))
proc.time() - start

sweden_X_train <- as.matrix(sweden_train[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')])
sweden_X_test <- as.matrix(sweden_test[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')])

sweden_enet <- glmnet(sweden_X_train, sweden_train$gpw, alpha=0, lambda=0.001)

sweden_enet_fitted <- as.numeric(predict(sweden_enet, newx=sweden_X_train, s=0.001))
mape(sweden_enet_fitted, sweden_train$gpw)
rmse(sweden_enet_fitted, sweden_train$gpw)
mae(sweden_enet_fitted, sweden_train$gpw)

sweden_enet_forecast <- as.numeric(predict(sweden_enet, newx=sweden_X_test, s=0.001))
mape(sweden_enet_forecast, sweden_test$gpw)
rmse(sweden_enet_forecast, sweden_test$gpw)
mae(sweden_enet_forecast, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main='sweden gpw Recycling - Elastic Net', col='black')
lines(y=sweden_enet_fitted, x=sweden_train$Year, col='red', type='b')
lines(y=sweden_enet_forecast, x=sweden_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#Cubist
start <- proc.time()
rolling_cv_cubist(sweden_train, c('GDP', 'co2', 'material_footprint', 'energy_consumption'))
proc.time() - start

sweden_cubist <- cubist(x=as.matrix(sweden_train[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')]),
                       y=sweden_train$gpw, committees=1)

sweden_cubist_fitted <- predict(sweden_cubist, 
                               newdata=as.data.frame(sweden_train[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')]),
                               neighbors=1)
mape(sweden_cubist_fitted, sweden_train$gpw)
rmse(sweden_cubist_fitted, sweden_train$gpw)
mae(sweden_cubist_fitted, sweden_train$gpw)

sweden_cubist_forecast <- predict(sweden_cubist,
                                 newdata=as.data.frame(sweden_test[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')]),
                                 neighbors=1)
mape(sweden_cubist_forecast, sweden_test$gpw)
rmse(sweden_cubist_forecast, sweden_test$gpw)
mae(sweden_cubist_forecast, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main='sweden gpw Recycling - Cubist', col='black')
lines(y=sweden_cubist_fitted, x=sweden_train$Year, col='red', type='b')
lines(y=sweden_cubist_forecast, x=sweden_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)


#Bayesian Ridge Regression
sweden_bayes_ridge <- bayesglm(gpw~GDP+co2+material_footprint+energy_consumption, data=sweden_train,
                              family=gaussian)

sweden_bayes_fitted <- predict(sweden_bayes_ridge, newdata=sweden_train)
mape(sweden_bayes_fitted, sweden_train$gpw)
rmse(sweden_bayes_fitted, sweden_train$gpw)
mae(sweden_bayes_fitted, sweden_train$gpw)

sweden_bayes_forecast <- predict(sweden_bayes_ridge, newdata=sweden_test)
mape(sweden_bayes_forecast, sweden_test$gpw)
rmse(sweden_bayes_forecast, sweden_test$gpw)
mae(sweden_bayes_forecast, sweden_test$gpw)

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main='sweden gpw Recycling - Bayesian Ridge', col='black')
lines(y=sweden_bayes_fitted, x=sweden_train$Year, col='red', type='b')
lines(y=sweden_bayes_forecast, x=sweden_test$Year, col='blue', type='b')
legend("topleft", legend=c("Actual", "Fitted", "Forecast"),
       col=c("black", "red", "blue"), lty=1, pch=1, bty="n", cex=0.75)

#plot----
pdf("glass_forecasts_plain_series.pdf", width=12, height=16)
par(mfrow=c(7, 2),
    mar=c(4, 5, 2, 1),
    cex.main=2,
    cex.lab=1.5,
    cex.axis=1.5)
plot(y=austria$gpw, x=austria$Year, type='b', xlab='Year',
     ylab='%', main='Austria Glass Packaging Recycling', col='black')

plot(y=belgium$gpw, x=belgium$Year, type='b', xlab='Year',
     ylab='%', main='Belgium Glass Packaging Recycling', col='black')

plot(y=denmark$gpw, x=denmark$Year, type='b', xlab='Year',
     ylab='%', main='Denmark Glass Packaging Recycling', col='black')

plot(y=finland$gpw, x=finland$Year, type='b', xlab='Year',
     ylab='%', main='Finland Glass Packaging Recycling', col='black')

plot(y=france$gpw, x=france$Year, type='b', xlab='Year',
     ylab='%', main='France Glass Packaging Recycling', col='black')

plot(y=germany$gpw, x=germany$Year, type='b', xlab='Year',
     ylab='%', main='Germany Glass Packaging Recycling', col='black')

plot(y=ireland$gpw, x=ireland$Year, type='b', xlab='Year',
     ylab='%', main='Ireland Glass Packaging Recycling', col='black')

plot(y=italy$gpw, x=italy$Year, type='b', xlab='Year',
     ylab='%', main='Italy Glass Packaging Recycling', col='black')

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b', xlab='Year',
     ylab='%', main='Luxembourg Glass Packaging Recycling', col='black')

plot(y=netherlands$gpw, x=netherlands$Year, type='b', xlab='Year',
     ylab='%', main='Netherlands Glass Packaging Recycling', col='black')

plot(y=portugal$gpw, x=portugal$Year, type='b', xlab='Year',
     ylab='%', main='Portugal Glass Packaging Recycling', col='black')

plot(y=spain$gpw, x=spain$Year, type='b', xlab='Year',
     ylab='%', main='Spain Glass Packaging Recycling', col='black')

plot(y=sweden$gpw, x=sweden$Year, type='b', xlab='Year',
     ylab='%', main='Sweden Glass Packaging Recycling', col='black')
dev.off()

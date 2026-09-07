#load packages----
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

#loading in the data----
#packaging data
pc_packaging_waste <- read_excel("Desktop/Packaging waste forecasting/Packaging Data/Paper_cardboard_packaging_waste.xlsx")
pc_packaging_waste$TIME <- as.double(pc_packaging_waste$TIME)

#training data
countries <- c("Austria", "Belgium", "Denmark", "Finland", "France", "Germany",
               "Ireland", "Italy", "Luxembourg", "Netherlands", "Portugal", "Spain", "Sweden")
vars <- c("Year", "Population", "GDP", "material_footprint", "energy_consumption", "co2", "exports")

modelling_data <- lapply(setNames(countries, tolower(countries)), function(ctry) {
  df <- read_excel(paste0("Desktop/Packaging waste forecasting/Country data/", ctry, ".xlsx")) |>
    dplyr::select(all_of(vars)) |>
    filter(Year >= 1997, Year <= 2022)
  df$pcpw <- pc_packaging_waste[[ctry]][pc_packaging_waste$TIME <= 2022]
  df
})
list2env(modelling_data, envir = .GlobalEnv)

portugal <- portugal[complete.cases(portugal), ] # as portugal is missing a year

#forecasting data
variables <- c("co2_forecasts", "energy_consumption_forecasts", "gdp_forecasts", 
               "exports_forecasts",  "material_footprint_forecasts", 
               "population_forecasts")

forecasting_data <- lapply(setNames(variables, tolower(variables)), function(ctry) {
  df <- read.csv(paste0("Desktop/Covariate Forecasting/Actually generating forecasts/", ctry, ".csv")) 
})
list2env(forecasting_data, envir = .GlobalEnv)

#fitting the models----
#austria
austria_bayes_ridge <- bayesglm(pcpw~material_footprint+exports, data=austria,
                                family=gaussian)

austria_train <- austria[austria$Year < 2018, ]
austria_test  <- austria[austria$Year >= 2018, ]

austria_bayes_ridge_test <- bayesglm(pcpw~material_footprint+exports, data=austria_train,
                                     family=gaussian)

austria_test_preds <- predict(austria_bayes_ridge_test,
                              newdata=austria_test[, c('material_footprint', 'exports')])
austria_test_residuals <- austria_test$pcpw - austria_test_preds
austria_centred_residuals <- austria_test_residuals - mean(austria_test_residuals)

austria_pc_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                   sim_id=rep(1:1000, each=8), value=NA)

set.seed(123)
for(i in 1:1000){
  austria_newdata <- data.frame(
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Austria'&material_footprint_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Austria'&exports_forecasts$sim_id==i,'value']
  )
  austria_forecast <- predict(austria_bayes_ridge, newdata=austria_newdata)
  austria_boot_residual <- sample(austria_centred_residuals, size=1)
  austria_pc_forecasts$value[austria_pc_forecasts$sim_id==i] <- austria_forecast + austria_boot_residual
}

austria_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  austria_median <- median(austria_pc_forecasts[austria_pc_forecasts$year==i,]$value)
  austria_lower  <- quantile(austria_pc_forecasts[austria_pc_forecasts$year==i,]$value, probs=0.025)
  austria_upper  <- quantile(austria_pc_forecasts[austria_pc_forecasts$year==i,]$value, probs=0.975)
  austria_median_pis[nrow(austria_median_pis)+1,] <- c(i, austria_median, austria_lower, austria_upper)
}


#belgium
belgium_svm <- svm(pcpw~exports+GDP, data=belgium, kernel='radial',
                   cost=3.1, gamma=0.05, epsilon=0.061)

belgium_train <- belgium[belgium$Year < 2018, ]
belgium_test  <- belgium[belgium$Year >= 2018, ]

belgium_svm_test <- svm(pcpw~exports+GDP, data=belgium_train, kernel='radial',
                        cost=3.1, gamma=0.05, epsilon=0.061)

belgium_test_preds <- predict(belgium_svm_test,
                              newdata=belgium_test[, c('exports', 'GDP')])
belgium_test_residuals <- belgium_test$pcpw - belgium_test_preds
belgium_centred_residuals <- belgium_test_residuals - mean(belgium_test_residuals)

belgium_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  belgium_newdata <- data.frame(
    GDP=gdp_forecasts[gdp_forecasts$country=='Belgium'&gdp_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Belgium'&exports_forecasts$sim_id==i,'value']
  )
  belgium_forecast <- predict(belgium_svm, newdata=belgium_newdata)
  belgium_boot_residual <- sample(belgium_centred_residuals, size=1)
  belgium_temp <- data.frame(year=2023:2030, sim_id=i, value=belgium_forecast + belgium_boot_residual)
  belgium_pc_forecasts <- rbind(belgium_pc_forecasts, belgium_temp)
}

belgium_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  belgium_median <- median(belgium_pc_forecasts[belgium_pc_forecasts$year==i,]$value)
  belgium_lower  <- quantile(belgium_pc_forecasts[belgium_pc_forecasts$year==i,]$value, probs=0.025)
  belgium_upper  <- quantile(belgium_pc_forecasts[belgium_pc_forecasts$year==i,]$value, probs=0.975)
  belgium_median_pis[nrow(belgium_median_pis)+1,] <- c(i, belgium_median, belgium_lower, belgium_upper)
}

#denmark
denmark_svm <- svm(pcpw~Population+co2+material_footprint, data=denmark, kernel='radial',
                   cost=5, gamma=0.1, epsilon=0.091)

denmark_train <- denmark[denmark$Year < 2018, ]
denmark_test  <- denmark[denmark$Year >= 2018, ]

denmark_svm_test <- svm(pcpw~Population+co2+material_footprint, data=denmark_train, kernel='radial',
                        cost=5, gamma=0.1, epsilon=0.091)

denmark_test_preds <- predict(denmark_svm_test,
                              newdata=denmark_test[, c('Population', 'co2', 'material_footprint')])
denmark_test_residuals <- denmark_test$pcpw - denmark_test_preds
denmark_centred_residuals <- denmark_test_residuals - mean(denmark_test_residuals)

denmark_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  denmark_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Denmark'&population_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Denmark'&co2_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Denmark'&material_footprint_forecasts$sim_id==i,'value']
  )
  denmark_forecast <- predict(denmark_svm, newdata=denmark_newdata)
  denmark_boot_residual <- sample(denmark_centred_residuals, size=1)
  denmark_temp <- data.frame(year=2023:2030, sim_id=i, value=denmark_forecast + denmark_boot_residual)
  denmark_pc_forecasts <- rbind(denmark_pc_forecasts, denmark_temp)
}

denmark_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  denmark_median <- median(denmark_pc_forecasts[denmark_pc_forecasts$year==i,]$value)
  denmark_lower  <- quantile(denmark_pc_forecasts[denmark_pc_forecasts$year==i,]$value, probs=0.025)
  denmark_upper  <- quantile(denmark_pc_forecasts[denmark_pc_forecasts$year==i,]$value, probs=0.975)
  denmark_median_pis[nrow(denmark_median_pis)+1,] <- c(i, denmark_median, denmark_lower, denmark_upper)
}

#finland
finland_svm <- svm(pcpw~Population+GDP, data=finland, kernel='radial',
                   cost=4.1, gamma=0.05, epsilon=0.091)

finland_train <- finland[finland$Year < 2018, ]
finland_test  <- finland[finland$Year >= 2018, ]

finland_svm_test <- svm(pcpw~Population+GDP, data=finland_train, kernel='radial',
                        cost=4.1, gamma=0.05, epsilon=0.091)

finland_test_preds <- predict(finland_svm_test,
                              newdata=finland_test[, c('Population', 'GDP')])
finland_test_residuals <- finland_test$pcpw - finland_test_preds
finland_centred_residuals <- finland_test_residuals - mean(finland_test_residuals)

finland_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  finland_newdata <- data.frame(
    GDP=gdp_forecasts[gdp_forecasts$country=='Finland'&gdp_forecasts$sim_id==i,'value'],
    Population=population_forecasts[population_forecasts$country=='Finland'&population_forecasts$sim_id==i,'value']
  )
  finland_forecast <- predict(finland_svm, newdata=finland_newdata)
  finland_boot_residual <- sample(finland_centred_residuals, size=1)
  finland_temp <- data.frame(year=2023:2030, sim_id=i, value=finland_forecast + finland_boot_residual)
  finland_pc_forecasts <- rbind(finland_pc_forecasts, finland_temp)
}

finland_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  finland_median <- median(finland_pc_forecasts[finland_pc_forecasts$year==i,]$value)
  finland_lower  <- quantile(finland_pc_forecasts[finland_pc_forecasts$year==i,]$value, probs=0.025)
  finland_upper  <- quantile(finland_pc_forecasts[finland_pc_forecasts$year==i,]$value, probs=0.975)
  finland_median_pis[nrow(finland_median_pis)+1,] <- c(i, finland_median, finland_lower, finland_upper)
}

#france
france_svm <- svm(pcpw~Population+material_footprint, data=france, kernel='radial',
                  cost=1.9, gamma=0.1, epsilon=0.091)

france_train <- france[france$Year < 2018, ]
france_test  <- france[france$Year >= 2018, ]

france_svm_test <- svm(pcpw~Population+material_footprint, data=france_train, kernel='radial',
                       cost=1.9, gamma=0.1, epsilon=0.091)

france_test_preds <- predict(france_svm_test,
                             newdata=france_test[, c('Population', 'material_footprint')])
france_test_residuals <- france_test$pcpw - france_test_preds
france_centred_residuals <- france_test_residuals - mean(france_test_residuals)

france_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  france_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='France'&population_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='France'&material_footprint_forecasts$sim_id==i,'value']
  )
  france_forecast <- predict(france_svm, newdata=france_newdata)
  france_boot_residual <- sample(france_centred_residuals, size=1)
  france_temp <- data.frame(year=2023:2030, sim_id=i, value=france_forecast + france_boot_residual)
  france_pc_forecasts <- rbind(france_pc_forecasts, france_temp)
}

france_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  france_median <- median(france_pc_forecasts[france_pc_forecasts$year==i,]$value)
  france_lower  <- quantile(france_pc_forecasts[france_pc_forecasts$year==i,]$value, probs=0.025)
  france_upper  <- quantile(france_pc_forecasts[france_pc_forecasts$year==i,]$value, probs=0.975)
  france_median_pis[nrow(france_median_pis)+1,] <- c(i, france_median, france_lower, france_upper)
}

#germany
germany_bayes_ridge <- bayesglm(pcpw~Population, data=germany, family=gaussian)

germany_train <- germany[germany$Year < 2018, ]
germany_test  <- germany[germany$Year >= 2018, ]

germany_bayes_ridge_test <- bayesglm(pcpw~Population, data=germany_train, family=gaussian)

germany_test_preds <- predict(germany_bayes_ridge_test,
                              newdata=germany_test[, 'Population', drop=FALSE])
germany_test_residuals <- germany_test$pcpw - germany_test_preds
germany_centred_residuals <- germany_test_residuals - mean(germany_test_residuals)

germany_pc_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                   sim_id=rep(1:1000, each=8), value=NA)

set.seed(123)
for(i in 1:1000){
  germany_newdata <- data.frame(Population=population_forecasts[population_forecasts$country=='Germany'&population_forecasts$sim_id==i,'value'])
  germany_forecast <- predict(germany_bayes_ridge, newdata=germany_newdata)
  germany_boot_residual <- sample(germany_centred_residuals, size=1)
  germany_pc_forecasts$value[germany_pc_forecasts$sim_id==i] <- germany_forecast + germany_boot_residual
}

germany_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  germany_median <- median(germany_pc_forecasts[germany_pc_forecasts$year==i,]$value)
  germany_lower  <- quantile(germany_pc_forecasts[germany_pc_forecasts$year==i,]$value, probs=0.025)
  germany_upper  <- quantile(germany_pc_forecasts[germany_pc_forecasts$year==i,]$value, probs=0.975)
  germany_median_pis[nrow(germany_median_pis)+1,] <- c(i, germany_median, germany_lower, germany_upper)
}

#ireland
ireland_svm <- svm(pcpw~Population+material_footprint+exports+energy_consumption, data=ireland, kernel='radial',
                   cost=1.5, gamma=0.1, epsilon=0.091)

ireland_train <- ireland[ireland$Year < 2018, ]
ireland_test  <- ireland[ireland$Year >= 2018, ]

ireland_svm_test <- svm(pcpw~Population+material_footprint+exports+energy_consumption, data=ireland_train, kernel='radial',
                        cost=1.5, gamma=0.1, epsilon=0.091)

ireland_test_preds <- predict(ireland_svm_test,
                              newdata=ireland_test[, c('Population', 'material_footprint', 'exports', 'energy_consumption')])
ireland_test_residuals <- ireland_test$pcpw - ireland_test_preds
ireland_centred_residuals <- ireland_test_residuals - mean(ireland_test_residuals)

ireland_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  ireland_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Ireland'&population_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Ireland'&material_footprint_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Ireland'&exports_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Ireland'&energy_consumption_forecasts$sim_id==i,'value']
  )
  ireland_forecast <- predict(ireland_svm, newdata=ireland_newdata)
  ireland_boot_residual <- sample(ireland_centred_residuals, size=1)
  ireland_temp <- data.frame(year=2023:2030, sim_id=i, value=ireland_forecast + ireland_boot_residual)
  ireland_pc_forecasts <- rbind(ireland_pc_forecasts, ireland_temp)
}

ireland_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  ireland_median <- median(ireland_pc_forecasts[ireland_pc_forecasts$year==i,]$value)
  ireland_lower  <- quantile(ireland_pc_forecasts[ireland_pc_forecasts$year==i,]$value, probs=0.025)
  ireland_upper  <- quantile(ireland_pc_forecasts[ireland_pc_forecasts$year==i,]$value, probs=0.975)
  ireland_median_pis[nrow(ireland_median_pis)+1,] <- c(i, ireland_median, ireland_lower, ireland_upper)
}

#italy
italy_xarima <- auto.arima(y=italy$pcpw, max.p=5, max.q=5,
                           xreg=as.matrix(italy[, c('Population', 'GDP', 'energy_consumption')]))

italy_train <- italy[italy$Year < 2018, ]
italy_test  <- italy[italy$Year >= 2018, ]

italy_xarima_test <- auto.arima(y=italy_train$pcpw, max.p=5, max.q=5,
                                xreg=as.matrix(italy_train[, c('Population', 'GDP', 'energy_consumption')]))

italy_test_preds <- as.numeric(predict(italy_xarima_test,
                                       newxreg=as.matrix(italy_test[, c('Population', 'GDP', 'energy_consumption')]),
                                       n.ahead=nrow(italy_test))$pred)
italy_test_residuals <- italy_test$pcpw - italy_test_preds
italy_centred_residuals <- italy_test_residuals - mean(italy_test_residuals)

italy_pc_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                 sim_id=rep(1:1000, each=8), value=NA)

set.seed(123)
for(i in 1:1000){
  italy_newxreg <- as.matrix(data.frame(
    Population=population_forecasts[population_forecasts$country=='Italy'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Italy'&gdp_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Italy'&energy_consumption_forecasts$sim_id==i,'value']
  ))
  italy_forecast <- as.numeric(predict(italy_xarima, newxreg=italy_newxreg, n.ahead=8)$pred)
  italy_boot_residual <- sample(italy_centred_residuals, size=1)
  italy_pc_forecasts$value[italy_pc_forecasts$sim_id==i] <- italy_forecast + italy_boot_residual
}

italy_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                               lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  italy_median <- median(italy_pc_forecasts[italy_pc_forecasts$year==i,]$value)
  italy_lower  <- quantile(italy_pc_forecasts[italy_pc_forecasts$year==i,]$value, probs=0.025)
  italy_upper  <- quantile(italy_pc_forecasts[italy_pc_forecasts$year==i,]$value, probs=0.975)
  italy_median_pis[nrow(italy_median_pis)+1,] <- c(i, italy_median, italy_lower, italy_upper)
}

#luxembourg
luxembourg_svm <- svm(pcpw~Population+GDP, data=luxembourg, kernel='radial',
                      cost=0.6, gamma=0.05, epsilon=0.091)

luxembourg_train <- luxembourg[luxembourg$Year < 2018, ]
luxembourg_test  <- luxembourg[luxembourg$Year >= 2018, ]

luxembourg_svm_test <- svm(pcpw~Population+GDP, data=luxembourg_train, kernel='radial',
                           cost=0.6, gamma=0.05, epsilon=0.091)

luxembourg_test_preds <- predict(luxembourg_svm_test,
                                 newdata=luxembourg_test[, c('Population', 'GDP')])
luxembourg_test_residuals <- luxembourg_test$pcpw - luxembourg_test_preds
luxembourg_centred_residuals <- luxembourg_test_residuals - mean(luxembourg_test_residuals)

luxembourg_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  luxembourg_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Luxembourg'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Luxembourg'&gdp_forecasts$sim_id==i,'value']
  )
  luxembourg_forecast <- predict(luxembourg_svm, newdata=luxembourg_newdata)
  luxembourg_boot_residual <- sample(luxembourg_centred_residuals, size=1)
  luxembourg_temp <- data.frame(year=2023:2030, sim_id=i, value=luxembourg_forecast + luxembourg_boot_residual)
  luxembourg_pc_forecasts <- rbind(luxembourg_pc_forecasts, luxembourg_temp)
}

luxembourg_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                    lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  luxembourg_median <- median(luxembourg_pc_forecasts[luxembourg_pc_forecasts$year==i,]$value)
  luxembourg_lower  <- quantile(luxembourg_pc_forecasts[luxembourg_pc_forecasts$year==i,]$value, probs=0.025)
  luxembourg_upper  <- quantile(luxembourg_pc_forecasts[luxembourg_pc_forecasts$year==i,]$value, probs=0.975)
  luxembourg_median_pis[nrow(luxembourg_median_pis)+1,] <- c(i, luxembourg_median, luxembourg_lower, luxembourg_upper)
}

#netherlands
netherlands_xarima <- auto.arima(y=netherlands$pcpw, max.p=5, max.q=5,
                                 xreg=as.matrix(netherlands[, c('GDP')]))

netherlands_train <- netherlands[netherlands$Year < 2018, ]
netherlands_test  <- netherlands[netherlands$Year >= 2018, ]

netherlands_xarima_test <- auto.arima(y=netherlands_train$pcpw, max.p=5, max.q=5,
                                      xreg=as.matrix(netherlands_train[, c('GDP')]))

netherlands_test_preds <- as.numeric(predict(netherlands_xarima_test,
                                             newxreg=as.matrix(netherlands_test[, c('GDP')]),
                                             n.ahead=nrow(netherlands_test))$pred)
netherlands_test_residuals <- netherlands_test$pcpw - netherlands_test_preds
netherlands_centred_residuals <- netherlands_test_residuals - mean(netherlands_test_residuals)

netherlands_pc_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                       sim_id=rep(1:1000, each=8), value=NA)

set.seed(123)
for(i in 1:1000){
  netherlands_newxreg <- as.matrix(data.frame(GDP=gdp_forecasts[gdp_forecasts$country=='Netherlands'&gdp_forecasts$sim_id==i,'value']))
  netherlands_forecast <- as.numeric(predict(netherlands_xarima, newxreg=netherlands_newxreg, n.ahead=8)$pred)
  netherlands_boot_residual <- sample(netherlands_centred_residuals, size=1)
  netherlands_pc_forecasts$value[netherlands_pc_forecasts$sim_id==i] <- netherlands_forecast + netherlands_boot_residual
}

netherlands_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                     lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  netherlands_median <- median(netherlands_pc_forecasts[netherlands_pc_forecasts$year==i,]$value)
  netherlands_lower  <- quantile(netherlands_pc_forecasts[netherlands_pc_forecasts$year==i,]$value, probs=0.025)
  netherlands_upper  <- quantile(netherlands_pc_forecasts[netherlands_pc_forecasts$year==i,]$value, probs=0.975)
  netherlands_median_pis[nrow(netherlands_median_pis)+1,] <- c(i, netherlands_median, netherlands_lower, netherlands_upper)
}

#portugal
portugal_svm <- svm(pcpw~Population+GDP+co2, data=portugal, kernel='radial',
                    cost=0.5, gamma=0.05, epsilon=0.011)

portugal_train <- portugal[portugal$Year < 2018, ]
portugal_test  <- portugal[portugal$Year >= 2018, ]

portugal_svm_test <- svm(pcpw~Population+GDP+co2, data=portugal_train, kernel='radial',
                         cost=0.5, gamma=0.05, epsilon=0.011)

portugal_test_preds <- predict(portugal_svm_test,
                               newdata=portugal_test[, c('Population', 'GDP', 'co2')])
portugal_test_residuals <- portugal_test$pcpw - portugal_test_preds
portugal_centred_residuals <- portugal_test_residuals - mean(portugal_test_residuals)

portugal_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  portugal_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Portugal'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Portugal'&gdp_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Portugal'&co2_forecasts$sim_id==i,'value']
  )
  portugal_forecast <- predict(portugal_svm, newdata=portugal_newdata)
  portugal_boot_residual <- sample(portugal_centred_residuals, size=1)
  portugal_temp <- data.frame(year=2023:2030, sim_id=i, value=portugal_forecast + portugal_boot_residual)
  portugal_pc_forecasts <- rbind(portugal_pc_forecasts, portugal_temp)
}

portugal_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                  lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  portugal_median <- median(portugal_pc_forecasts[portugal_pc_forecasts$year==i,]$value)
  portugal_lower  <- quantile(portugal_pc_forecasts[portugal_pc_forecasts$year==i,]$value, probs=0.025)
  portugal_upper  <- quantile(portugal_pc_forecasts[portugal_pc_forecasts$year==i,]$value, probs=0.975)
  portugal_median_pis[nrow(portugal_median_pis)+1,] <- c(i, portugal_median, portugal_lower, portugal_upper)
}

#spain
spain_svm <- svm(pcpw~Population+GDP, data=spain, kernel='radial',
                 cost=4.7, gamma=0.1, epsilon=0.001)

spain_train <- spain[spain$Year < 2018, ]
spain_test  <- spain[spain$Year >= 2018, ]

spain_svm_test <- svm(pcpw~Population+GDP, data=spain_train, kernel='radial',
                      cost=4.7, gamma=0.1, epsilon=0.001)

spain_test_preds <- predict(spain_svm_test,
                            newdata=spain_test[, c('Population', 'GDP')])
spain_test_residuals <- spain_test$pcpw - spain_test_preds
spain_centred_residuals <- spain_test_residuals - mean(spain_test_residuals)

spain_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  spain_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Spain'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Spain'&gdp_forecasts$sim_id==i,'value']
  )
  spain_forecast <- predict(spain_svm, newdata=spain_newdata)
  spain_boot_residual <- sample(spain_centred_residuals, size=1)
  spain_temp <- data.frame(year=2023:2030, sim_id=i, value=spain_forecast + spain_boot_residual)
  spain_pc_forecasts <- rbind(spain_pc_forecasts, spain_temp)
}

spain_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                               lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  spain_median <- median(spain_pc_forecasts[spain_pc_forecasts$year==i,]$value)
  spain_lower  <- quantile(spain_pc_forecasts[spain_pc_forecasts$year==i,]$value, probs=0.025)
  spain_upper  <- quantile(spain_pc_forecasts[spain_pc_forecasts$year==i,]$value, probs=0.975)
  spain_median_pis[nrow(spain_median_pis)+1,] <- c(i, spain_median, spain_lower, spain_upper)
}

#sweden
sweden_svm <- svm(pcpw~Population, data=sweden, kernel='radial',
                  cost=5, gamma=0.5, epsilon=0.051)

sweden_train <- sweden[sweden$Year < 2018, ]
sweden_test  <- sweden[sweden$Year >= 2018, ]

sweden_svm_test <- svm(pcpw~Population, data=sweden_train, kernel='radial',
                       cost=5, gamma=0.5, epsilon=0.051)

sweden_test_preds <- predict(sweden_svm_test,
                             newdata=sweden_test[, 'Population', drop=FALSE])
sweden_test_residuals <- sweden_test$pcpw - sweden_test_preds
sweden_centred_residuals <- sweden_test_residuals - mean(sweden_test_residuals)

sweden_pc_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  sweden_newdata <- data.frame(Population=population_forecasts[population_forecasts$country=='Sweden'&population_forecasts$sim_id==i,'value'])
  sweden_forecast <- predict(sweden_svm, newdata=sweden_newdata)
  sweden_boot_residual <- sample(sweden_centred_residuals, size=1)
  sweden_temp <- data.frame(year=2023:2030, sim_id=i, value=sweden_forecast + sweden_boot_residual)
  sweden_pc_forecasts <- rbind(sweden_pc_forecasts, sweden_temp)
}

sweden_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  sweden_median <- median(sweden_pc_forecasts[sweden_pc_forecasts$year==i,]$value)
  sweden_lower  <- quantile(sweden_pc_forecasts[sweden_pc_forecasts$year==i,]$value, probs=0.025)
  sweden_upper  <- quantile(sweden_pc_forecasts[sweden_pc_forecasts$year==i,]$value, probs=0.975)
  sweden_median_pis[nrow(sweden_median_pis)+1,] <- c(i, sweden_median, sweden_lower, sweden_upper)
}


#set up the dataframe----
pc_meta <- data.frame(country=c("Austria", "Belgium", "Denmark", "Finland", 
                                      "France", "Germany","Ireland", "Italy", "Luxembourg", 
                                      "Netherlands", "Portugal", "Spain", "Sweden"),
                            stream="paper_cardboard",
                            model=c('BRR', 'SVM', 'SVM', 'SVM', 'SVM', 'BRR', 'SVM', 
                                    'XARIMA', 'SVM', 'XARIMA', 'SVM', 'SVM', 'SVM'),
                            n_covariates=c(2, 2, 3, 2, 2, 1, 4, 3, 2, 1, 3, 2, 1)
)

#2023 forcast error & absolute forecast error----
pc_meta$forecast_error_2023 <- c(pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Austria-austria_median_pis[austria_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Belgium-belgium_median_pis[belgium_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Denmark-denmark_median_pis[denmark_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Finland-finland_median_pis[finland_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$France-france_median_pis[france_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Germany-germany_median_pis[germany_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Ireland-ireland_median_pis[ireland_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Italy-italy_median_pis[italy_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Luxembourg-luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Netherlands-netherlands_median_pis[netherlands_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Portugal-portugal_median_pis[portugal_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Spain-spain_median_pis[spain_median_pis$year=='2023',]$median_forecast,
                                       pc_packaging_waste[pc_packaging_waste$TIME=='2023',]$Sweden-sweden_median_pis[sweden_median_pis$year=='2023',]$median_forecast
)

pc_meta$abs_forecast_error_2023 <- abs(pc_meta$forecast_error_2023)

#PI width----
pc_meta$pi_width_2023 <- c(austria_median_pis[austria_median_pis$year=='2023',]$upper_pi-austria_median_pis[austria_median_pis$year=='2023',]$lower_pi,
                                 belgium_median_pis[belgium_median_pis$year=='2023',]$upper_pi-belgium_median_pis[belgium_median_pis$year=='2023',]$lower_pi,
                                 denmark_median_pis[denmark_median_pis$year=='2023',]$upper_pi-denmark_median_pis[denmark_median_pis$year=='2023',]$lower_pi,
                                 finland_median_pis[finland_median_pis$year=='2023',]$upper_pi-finland_median_pis[finland_median_pis$year=='2023',]$lower_pi,
                                 france_median_pis[france_median_pis$year=='2023',]$upper_pi-france_median_pis[france_median_pis$year=='2023',]$lower_pi,
                                 germany_median_pis[germany_median_pis$year=='2023',]$upper_pi-germany_median_pis[germany_median_pis$year=='2023',]$lower_pi,
                                 ireland_median_pis[ireland_median_pis$year=='2023',]$upper_pi-ireland_median_pis[ireland_median_pis$year=='2023',]$lower_pi,
                                 italy_median_pis[italy_median_pis$year=='2023',]$upper_pi-italy_median_pis[italy_median_pis$year=='2023',]$lower_pi,
                                 luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$upper_pi-luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$lower_pi,
                                 netherlands_median_pis[netherlands_median_pis$year=='2023',]$upper_pi-netherlands_median_pis[netherlands_median_pis$year=='2023',]$lower_pi,
                                 portugal_median_pis[portugal_median_pis$year=='2023',]$upper_pi-portugal_median_pis[portugal_median_pis$year=='2023',]$lower_pi,
                                 spain_median_pis[spain_median_pis$year=='2023',]$upper_pi-spain_median_pis[spain_median_pis$year=='2023',]$lower_pi,
                                 sweden_median_pis[sweden_median_pis$year=='2023',]$upper_pi-sweden_median_pis[sweden_median_pis$year=='2023',]$lower_pi
)

#training standard deviation, test standard deviation, ratio----
pc_meta$train_sd <- c(sd(austria[austria$Year<=2017,]$pcpw),
                            sd(belgium[belgium$Year<=2017,]$pcpw),
                            sd(denmark[denmark$Year<=2017,]$pcpw),
                            sd(finland[finland$Year<=2017,]$pcpw),
                            sd(france[france$Year<=2017,]$pcpw),
                            sd(germany[germany$Year<=2017,]$pcpw),
                            sd(ireland[ireland$Year<=2017,]$pcpw),
                            sd(italy[italy$Year<=2017,]$pcpw),
                            sd(luxembourg[luxembourg$Year<=2017,]$pcpw),
                            sd(netherlands[netherlands$Year<=2017,]$pcpw),
                            sd(portugal[portugal$Year<=2017,]$pcpw),
                            sd(spain[spain$Year<=2017,]$pcpw),
                            sd(sweden[sweden$Year<=2017,]$pcpw)
)

pc_meta$test_sd <- c(sd(austria[austria$Year>=2018,]$pcpw),
                           sd(belgium[belgium$Year>=2018,]$pcpw),
                           sd(denmark[denmark$Year>=2018,]$pcpw),
                           sd(finland[finland$Year>=2018,]$pcpw),
                           sd(france[france$Year>=2018,]$pcpw),
                           sd(germany[germany$Year>=2018,]$pcpw),
                           sd(ireland[ireland$Year>=2018,]$pcpw),
                           sd(italy[italy$Year>=2018,]$pcpw),
                           sd(luxembourg[luxembourg$Year>=2018,]$pcpw),
                           sd(netherlands[netherlands$Year>=2018,]$pcpw),
                           sd(portugal[portugal$Year>=2018,]$pcpw),
                           sd(spain[spain$Year>=2018,]$pcpw),
                           sd(sweden[sweden$Year>=2018,]$pcpw)
)

pc_meta$sd_ratio <- pc_meta$test_sd/pc_meta$train_sd

#trend of training set, trend of whole series, their difference, binary change of trend direction----
trend_slope <- function(df, year_col = "Year", y_col = "pcpw"){
  coef(lm(df[[y_col]] ~ df[[year_col]]))[2]
}

pc_meta$slope_full <- c(trend_slope(austria), trend_slope(belgium),
                              trend_slope(denmark), trend_slope(finland),
                              trend_slope(france), trend_slope(germany),
                              trend_slope(ireland), trend_slope(italy),
                              trend_slope(luxembourg), trend_slope(netherlands),
                              trend_slope(portugal), trend_slope(spain),
                              trend_slope(sweden)
)

pc_meta$slope_train <- c(
  trend_slope(austria[austria$Year<=2017,]), trend_slope(belgium[belgium$Year<=2017,]),
  trend_slope(denmark[denmark$Year<=2017,]), trend_slope(finland[finland$Year<=2017,]),
  trend_slope(france[france$Year<=2017,]), trend_slope(germany[germany$Year<=2017,]),
  trend_slope(ireland[ireland$Year<=2017,]), trend_slope(italy[italy$Year<=2017,]),
  trend_slope(luxembourg[luxembourg$Year<=2017,]), trend_slope(netherlands[netherlands$Year<=2017,]),
  trend_slope(portugal[portugal$Year<=2017,]), trend_slope(spain[spain$Year<=2017,]),
  trend_slope(sweden[sweden$Year<=2017,])
)

pc_meta$slope_diff <- pc_meta$slope_full - pc_meta$slope_train


pc_meta$trend_reversal <- as.integer(sign(pc_meta$slope_full)!=sign(pc_meta$slope_train))

#maximum and mean absolute change (Rise/fall) over the whole series----
max_abs_change <- function(df, y_col = "pcpw"){
  max(abs(diff(df[[y_col]])))
}

mean_abs_change <- function(df, y_col = "pcpw"){
  mean(abs(diff(df[[y_col]])))
}

pc_meta$max_abs_change <- c(max_abs_change(austria), max_abs_change(belgium),
                                  max_abs_change(denmark), max_abs_change(finland),
                                  max_abs_change(france), max_abs_change(germany),
                                  max_abs_change(ireland), max_abs_change(italy),
                                  max_abs_change(luxembourg), max_abs_change(netherlands),
                                  max_abs_change(portugal), max_abs_change(spain),
                                  max_abs_change(sweden)
)

pc_meta$mean_abs_change <- c(mean_abs_change(austria), mean_abs_change(belgium),
                                   mean_abs_change(denmark), mean_abs_change(finland),
                                   mean_abs_change(france), mean_abs_change(germany),
                                   mean_abs_change(ireland), mean_abs_change(italy), 
                                   mean_abs_change(luxembourg), mean_abs_change(netherlands),
                                   mean_abs_change(portugal), mean_abs_change(spain),
                                   mean_abs_change(sweden)
)

#writing the csv file----
write.csv(pc_meta, file='paper_cardboard_meta_dataset.csv')


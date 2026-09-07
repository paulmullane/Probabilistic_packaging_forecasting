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
plastic_packaging_waste <- read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Packaging waste forecasting/Packaging Data/Plastic_Packaging waste.xlsx")
plastic_packaging_waste$TIME <- as.double(plastic_packaging_waste$TIME)

#training data
countries <- c("Austria", "Belgium", "Denmark", "Finland", "France", "Germany",
               "Ireland", "Italy", "Luxembourg", "Netherlands", "Portugal", "Spain", "Sweden")
vars <- c("Year", "Population", "GDP", "material_footprint", "energy_consumption", "co2", "exports")

modelling_data <- lapply(setNames(countries, tolower(countries)), function(ctry) {
  df <- read_excel(paste0("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Packaging waste forecasting/Country data/", ctry, ".xlsx")) |>
    dplyr::select(all_of(vars)) |>
    filter(Year >= 1997, Year <= 2022)
  df$ppw <- plastic_packaging_waste[[ctry]][plastic_packaging_waste$TIME <= 2022]
  df
})
list2env(modelling_data, envir = .GlobalEnv)

portugal <- portugal[complete.cases(portugal), ] # as portugal is missing a year

#forecasting data
variables <- c("co2_forecasts", "energy_consumption_forecasts", "gdp_forecasts", 
               "exports_forecasts",  "material_footprint_forecasts", 
               "population_forecasts")

forecasting_data <- lapply(setNames(variables, tolower(variables)), function(ctry) {
  df <- read.csv(paste0("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Actually generating forecasts/", ctry, ".csv")) 
})
list2env(forecasting_data, envir = .GlobalEnv)

#fitting models----
#austria
austria_svm <- svm(ppw~GDP+exports+co2, data=austria, kernel='radial',
                   cost=0.7, gamma=0.15, epsilon=0.001)

# refit on pre-2018 data to compute genuine test set residuals
austria_train <- austria[austria$Year < 2018, ]
austria_test  <- austria[austria$Year >= 2018, ]

austria_svm_test <- svm(ppw~GDP+exports+co2, data=austria_train, kernel='radial',
                        cost=0.7, gamma=0.15, epsilon=0.001)

austria_test_preds <- predict(austria_svm_test,
                              newdata=austria_test[, c('GDP', 'exports', 'co2')])
austria_test_residuals <- austria_test$ppw - austria_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
austria_centred_residuals <- austria_test_residuals - mean(austria_test_residuals)

# blank dataframe to store the results
austria_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

# iterate over the sim ids
set.seed(123)
for (i in 1:1000){
  austria_newdata <- data.frame(
    GDP=gdp_forecasts[gdp_forecasts$country=='Austria'&gdp_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Austria'&exports_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Austria'&co2_forecasts$sim_id==i,'value']
  )
  austria_forecast <- predict(austria_svm, newdata=austria_newdata)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  austria_boot_residual <- sample(austria_centred_residuals, size=1)
  austria_forecast_with_residuals <- austria_forecast + austria_boot_residual
  
  austria_temp <- data.frame(year=2023:2030, sim_id=i,
                             value=austria_forecast_with_residuals)
  austria_plastic_forecasts <- rbind(austria_plastic_forecasts, austria_temp)
}

# blank dataframe for median forecast and PIs
austria_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  austria_median <- median(austria_plastic_forecasts[austria_plastic_forecasts$year==i,]$value)
  austria_lower  <- quantile(austria_plastic_forecasts[austria_plastic_forecasts$year==i,]$value, probs=0.025)
  austria_upper  <- quantile(austria_plastic_forecasts[austria_plastic_forecasts$year==i,]$value, probs=0.975)
  austria_median_pis[nrow(austria_median_pis)+1,] <- c(i, austria_median, austria_lower, austria_upper)
}

#belgium
belgium_xarima <- auto.arima(y=belgium$ppw, max.p=5, max.q=5,
                             xreg=as.matrix(belgium[, c('GDP', 'Population')]))

# refit on pre-2018 data to compute genuine test set residuals
belgium_train <- belgium[belgium$Year < 2018, ]
belgium_test  <- belgium[belgium$Year >= 2018, ]

belgium_xarima_test <- auto.arima(y=belgium_train$ppw, max.p=5, max.q=5,
                                  xreg=as.matrix(belgium_train[, c('GDP', 'Population')]))

belgium_test_preds <- as.numeric(predict(belgium_xarima_test,
                                         newxreg=as.matrix(belgium_test[, c('GDP', 'Population')]),
                                         n.ahead=nrow(belgium_test))$pred)
belgium_test_residuals <- belgium_test$ppw - belgium_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
belgium_centred_residuals <- belgium_test_residuals - mean(belgium_test_residuals)

# blank dataframe to store the results
belgium_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

# iterate over the sim ids
set.seed(123)
for (i in 1:1000){
  belgium_newdata <- data.frame(
    GDP=gdp_forecasts[gdp_forecasts$country=='Belgium'&gdp_forecasts$sim_id==i,'value'],
    Population=population_forecasts[population_forecasts$country=='Belgium'&population_forecasts$sim_id==i,'value']
  )
  belgium_forecast <- as.numeric(predict(belgium_xarima,
                                         newxreg=as.matrix(belgium_newdata[, c('GDP', 'Population')]),
                                         n.ahead=8)$pred)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  belgium_boot_residual <- sample(belgium_centred_residuals, size=1)
  belgium_forecast_with_residuals <- belgium_forecast + belgium_boot_residual
  
  belgium_temp <- data.frame(year=2023:2030, sim_id=i,
                             value=belgium_forecast_with_residuals)
  belgium_plastic_forecasts <- rbind(belgium_plastic_forecasts, belgium_temp)
}

# blank dataframe for median forecast and PIs
belgium_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  belgium_median <- median(belgium_plastic_forecasts[belgium_plastic_forecasts$year==i,]$value)
  belgium_lower  <- quantile(belgium_plastic_forecasts[belgium_plastic_forecasts$year==i,]$value, probs=0.025)
  belgium_upper  <- quantile(belgium_plastic_forecasts[belgium_plastic_forecasts$year==i,]$value, probs=0.975)
  belgium_median_pis[nrow(belgium_median_pis)+1,] <- c(i, belgium_median, belgium_lower, belgium_upper)
}

#denmark
denmark_svm <- svm(ppw~Population+GDP, data=denmark, kernel='radial',
                   cost=4.9, gamma=0.05, epsilon=0.091)

# refit on pre-2018 data to compute genuine test set residuals
denmark_train <- denmark[denmark$Year < 2018, ]
denmark_test  <- denmark[denmark$Year >= 2018, ]

denmark_svm_test <- svm(ppw~Population+GDP, data=denmark_train, kernel='radial',
                        cost=4.9, gamma=0.05, epsilon=0.091)

denmark_test_preds <- predict(denmark_svm_test,
                              newdata=denmark_test[, c('Population', 'GDP')])
denmark_test_residuals <- denmark_test$ppw - denmark_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
denmark_centred_residuals <- denmark_test_residuals - mean(denmark_test_residuals)

# blank dataframe to store the results
denmark_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

# iterate over the sim ids
set.seed(123)
for (i in 1:1000){
  denmark_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Denmark'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Denmark'&gdp_forecasts$sim_id==i,'value']
  )
  denmark_forecast <- predict(denmark_svm, newdata=denmark_newdata)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  denmark_boot_residual <- sample(denmark_centred_residuals, size=1)
  denmark_forecast_with_residuals <- denmark_forecast + denmark_boot_residual
  
  denmark_temp <- data.frame(year=2023:2030, sim_id=i,
                             value=denmark_forecast_with_residuals)
  denmark_plastic_forecasts <- rbind(denmark_plastic_forecasts, denmark_temp)
}

# blank dataframe for median forecast and PIs
denmark_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  denmark_median <- median(denmark_plastic_forecasts[denmark_plastic_forecasts$year==i,]$value)
  denmark_lower  <- quantile(denmark_plastic_forecasts[denmark_plastic_forecasts$year==i,]$value, probs=0.025)
  denmark_upper  <- quantile(denmark_plastic_forecasts[denmark_plastic_forecasts$year==i,]$value, probs=0.975)
  denmark_median_pis[nrow(denmark_median_pis)+1,] <- c(i, denmark_median, denmark_lower, denmark_upper)
}

#finland
set.seed(123)
finland_bayes_ridge <- bayesglm(ppw~Population+GDP, data=finland,
                                family=gaussian)

# refit on pre-2018 data to compute genuine test set residuals
finland_train <- finland[finland$Year < 2018, ]
finland_test  <- finland[finland$Year >= 2018, ]

finland_bayes_ridge_test <- bayesglm(ppw~Population+GDP, data=finland_train,
                                     family=gaussian)

finland_test_preds <- predict(finland_bayes_ridge_test,
                              newdata=finland_test[, c('Population', 'GDP')])
finland_test_residuals <- finland_test$ppw - finland_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
finland_centred_residuals <- finland_test_residuals - mean(finland_test_residuals)

# blank dataframe to store the results
finland_plastic_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                        sim_id=rep(1:1000, each=8), value=NA)

# iterate over the sim ids
for(i in 1:1000){
  finland_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Finland'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Finland'&gdp_forecasts$sim_id==i,'value']
  )
  finland_forecast <- predict(finland_bayes_ridge, newdata=finland_newdata)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  finland_boot_residual <- sample(finland_centred_residuals, size=1)
  finland_plastic_forecasts$value[finland_plastic_forecasts$sim_id==i] <- finland_forecast + finland_boot_residual
}

# blank dataframe for median forecast and PIs
finland_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  finland_median <- median(finland_plastic_forecasts[finland_plastic_forecasts$year==i,]$value)
  finland_lower  <- quantile(finland_plastic_forecasts[finland_plastic_forecasts$year==i,]$value, probs=0.025)
  finland_upper  <- quantile(finland_plastic_forecasts[finland_plastic_forecasts$year==i,]$value, probs=0.975)
  finland_median_pis[nrow(finland_median_pis)+1,] <- c(i, finland_median, finland_lower, finland_upper)
}

#france
france_svm <- svm(ppw~Population+GDP+exports+energy_consumption, data=france, kernel='radial',
                  cost=5, gamma=0.05, epsilon=0.011)

# refit on pre-2018 data to compute genuine test set residuals
france_train <- france[france$Year < 2018, ]
france_test  <- france[france$Year >= 2018, ]

france_svm_test <- svm(ppw~Population+GDP+exports+energy_consumption, data=france_train, kernel='radial',
                       cost=5, gamma=0.05, epsilon=0.011)

france_test_preds <- predict(france_svm_test,
                             newdata=france_test[, c('Population', 'GDP', 'exports', 'energy_consumption')])
france_test_residuals <- france_test$ppw - france_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
france_centred_residuals <- france_test_residuals - mean(france_test_residuals)

# blank dataframe to store the results
france_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

# iterate over the sim ids
set.seed(123)
for (i in 1:1000){
  france_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='France'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='France'&gdp_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='France'&exports_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='France'&energy_consumption_forecasts$sim_id==i,'value']
  )
  france_forecast <- predict(france_svm, newdata=france_newdata)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  france_boot_residual <- sample(france_centred_residuals, size=1)
  france_forecast_with_residuals <- france_forecast + france_boot_residual
  
  france_temp <- data.frame(year=2023:2030, sim_id=i,
                            value=france_forecast_with_residuals)
  france_plastic_forecasts <- rbind(france_plastic_forecasts, france_temp)
}

# blank dataframe for median forecast and PIs
france_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  france_median <- median(france_plastic_forecasts[france_plastic_forecasts$year==i,]$value)
  france_lower  <- quantile(france_plastic_forecasts[france_plastic_forecasts$year==i,]$value, probs=0.025)
  france_upper  <- quantile(france_plastic_forecasts[france_plastic_forecasts$year==i,]$value, probs=0.975)
  france_median_pis[nrow(france_median_pis)+1,] <- c(i, france_median, france_lower, france_upper)
}

#germany
germany_svm <- svm(ppw~Population+GDP+exports, data=germany, kernel='radial',
                   cost=3.5, gamma=0, epsilon=0.001)

# refit on pre-2018 data to compute genuine test set residuals
germany_train <- germany[germany$Year < 2018, ]
germany_test  <- germany[germany$Year >= 2018, ]

germany_svm_test <- svm(ppw~Population+GDP+exports, data=germany_train, kernel='radial',
                        cost=3.5, gamma=0, epsilon=0.001)

germany_test_preds <- predict(germany_svm_test,
                              newdata=germany_test[, c('Population', 'GDP', 'exports')])
germany_test_residuals <- germany_test$ppw - germany_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
germany_centred_residuals <- germany_test_residuals - mean(germany_test_residuals)

# blank dataframe to store the results
germany_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

# iterate over the sim ids
set.seed(123)
for(i in 1:1000){
  germany_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Germany'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Germany'&gdp_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Germany'&exports_forecasts$sim_id==i,'value']
  )
  germany_forecast <- predict(germany_svm, newdata=germany_newdata)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  germany_boot_residual <- sample(germany_centred_residuals, size=1)
  germany_forecast_with_residuals <- germany_forecast + germany_boot_residual
  
  germany_temp <- data.frame(year=2023:2030, sim_id=i,
                             value=germany_forecast_with_residuals)
  germany_plastic_forecasts <- rbind(germany_plastic_forecasts, germany_temp)
}

# blank dataframe for median forecast and PIs
germany_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  germany_median <- median(germany_plastic_forecasts[germany_plastic_forecasts$year==i,]$value)
  germany_lower  <- quantile(germany_plastic_forecasts[germany_plastic_forecasts$year==i,]$value, probs=0.025)
  germany_upper  <- quantile(germany_plastic_forecasts[germany_plastic_forecasts$year==i,]$value, probs=0.975)
  germany_median_pis[nrow(germany_median_pis)+1,] <- c(i, germany_median, germany_lower, germany_upper)
}

#ireland
set.seed(20229798)
ireland_svm <- svm(ppw~Population+GDP+co2+material_footprint+exports+energy_consumption, data=ireland, kernel='radial',
                   cost=2, gamma=0.05, epsilon=0.091)

# refit on pre-2018 data to compute genuine test set residuals
ireland_train <- ireland[ireland$Year < 2018, ]
ireland_test  <- ireland[ireland$Year >= 2018, ]

set.seed(20229798)
ireland_svm_test <- svm(ppw~Population+GDP+co2+material_footprint+exports+energy_consumption, data=ireland_train, kernel='radial',
                        cost=2, gamma=0.05, epsilon=0.091)

ireland_test_preds <- predict(ireland_svm_test,
                              newdata=ireland_test[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')])
ireland_test_residuals <- ireland_test$ppw - ireland_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
ireland_centred_residuals <- ireland_test_residuals - mean(ireland_test_residuals)

# blank dataframe to store the results
ireland_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

# iterate over the sim ids
set.seed(20229798)
for (i in 1:1000){
  ireland_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Ireland'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Ireland'&gdp_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Ireland'&co2_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Ireland'&material_footprint_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Ireland'&exports_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Ireland'&energy_consumption_forecasts$sim_id==i,'value']
  )
  ireland_forecast <- predict(ireland_svm, newdata=ireland_newdata)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  ireland_boot_residual <- sample(ireland_centred_residuals, size=1)
  ireland_forecast_with_residuals <- ireland_forecast + ireland_boot_residual
  
  ireland_temp <- data.frame(year=2023:2030, sim_id=i,
                             value=ireland_forecast_with_residuals)
  ireland_plastic_forecasts <- rbind(ireland_plastic_forecasts, ireland_temp)
}

# blank dataframe for median forecast and PIs
ireland_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  ireland_median <- median(ireland_plastic_forecasts[ireland_plastic_forecasts$year==i,]$value)
  ireland_lower  <- quantile(ireland_plastic_forecasts[ireland_plastic_forecasts$year==i,]$value, probs=0.025)
  ireland_upper  <- quantile(ireland_plastic_forecasts[ireland_plastic_forecasts$year==i,]$value, probs=0.975)
  ireland_median_pis[nrow(ireland_median_pis)+1,] <- c(i, ireland_median, ireland_lower, ireland_upper)
}

#italy
italy_bayes_ridge <- bayesglm(ppw~GDP+Population+co2+energy_consumption, data=italy,
                              family=gaussian)

# refit on pre-2018 data to compute genuine test set residuals
italy_train <- italy[italy$Year < 2018, ]
italy_test  <- italy[italy$Year >= 2018, ]

italy_bayes_ridge_test <- bayesglm(ppw~GDP+Population+co2+energy_consumption, data=italy_train,
                                   family=gaussian)

italy_test_preds <- predict(italy_bayes_ridge_test,
                            newdata=italy_test[, c('GDP', 'Population', 'co2', 'energy_consumption')])
italy_test_residuals <- italy_test$ppw - italy_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
italy_centred_residuals <- italy_test_residuals - mean(italy_test_residuals)

# blank dataframe to store the results
italy_plastic_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                      sim_id=rep(1:1000, each=8), value=NA)

# iterate over the sim ids
set.seed(123)
for (i in 1:1000){
  italy_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Italy'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Italy'&gdp_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Italy'&co2_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Italy'&energy_consumption_forecasts$sim_id==i,'value']
  )
  italy_forecast <- predict(italy_bayes_ridge, newdata=italy_newdata)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  italy_boot_residual <- sample(italy_centred_residuals, size=1)
  italy_plastic_forecasts$value[italy_plastic_forecasts$sim_id==i] <- italy_forecast + italy_boot_residual
}

# blank dataframe for median forecast and PIs
italy_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                               lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  italy_median <- median(italy_plastic_forecasts[italy_plastic_forecasts$year==i,]$value)
  italy_lower  <- quantile(italy_plastic_forecasts[italy_plastic_forecasts$year==i,]$value, probs=0.025)
  italy_upper  <- quantile(italy_plastic_forecasts[italy_plastic_forecasts$year==i,]$value, probs=0.975)
  italy_median_pis[nrow(italy_median_pis)+1,] <- c(i, italy_median, italy_lower, italy_upper)
}

#luxembourg
luxembourg_xarima <- auto.arima(y=luxembourg$ppw, max.p=5, max.q=5,
                                xreg=as.matrix(luxembourg[, c('Population', 'GDP', 'material_footprint', 'exports', 'energy_consumption')]))

# refit on pre-2018 data to compute genuine test set residuals
luxembourg_train <- luxembourg[luxembourg$Year < 2018, ]
luxembourg_test  <- luxembourg[luxembourg$Year >= 2018, ]

luxembourg_xarima_test <- auto.arima(y=luxembourg_train$ppw, max.p=5, max.q=5,
                                     xreg=as.matrix(luxembourg_train[, c('Population', 'GDP', 'material_footprint', 'exports', 'energy_consumption')]))

luxembourg_test_preds <- as.numeric(predict(luxembourg_xarima_test,
                                            newxreg=as.matrix(luxembourg_test[, c('Population', 'GDP', 'material_footprint', 'exports', 'energy_consumption')]),
                                            n.ahead=nrow(luxembourg_test))$pred)
luxembourg_test_residuals <- luxembourg_test$ppw - luxembourg_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
luxembourg_centred_residuals <- luxembourg_test_residuals - mean(luxembourg_test_residuals)

# blank dataframe to store the results
luxembourg_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

# iterate over the sim ids
set.seed(123)
for (i in 1:1000){
  luxembourg_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Luxembourg'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Luxembourg'&gdp_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Luxembourg'&material_footprint_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Luxembourg'&exports_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Luxembourg'&energy_consumption_forecasts$sim_id==i,'value']
  )
  luxembourg_forecast <- as.numeric(predict(luxembourg_xarima,
                                            newxreg=as.matrix(luxembourg_newdata[, c('Population', 'GDP', 'material_footprint', 'exports', 'energy_consumption')]),
                                            n.ahead=8)$pred)
  
  # sample ONE centred residual per simulation, applied across all forecast years
  luxembourg_boot_residual <- sample(luxembourg_centred_residuals, size=1)
  luxembourg_forecast_with_residuals <- luxembourg_forecast + luxembourg_boot_residual
  
  luxembourg_temp <- data.frame(year=2023:2030, sim_id=i,
                                value=luxembourg_forecast_with_residuals)
  luxembourg_plastic_forecasts <- rbind(luxembourg_plastic_forecasts, luxembourg_temp)
}

# blank dataframe for median forecast and PIs
luxembourg_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                    lower_pi=numeric(), upper_pi=numeric())

# calculate median forecast and PI values for each year
for(i in 2023:2030){
  luxembourg_median <- median(luxembourg_plastic_forecasts[luxembourg_plastic_forecasts$year==i,]$value)
  luxembourg_lower  <- quantile(luxembourg_plastic_forecasts[luxembourg_plastic_forecasts$year==i,]$value, probs=0.025)
  luxembourg_upper  <- quantile(luxembourg_plastic_forecasts[luxembourg_plastic_forecasts$year==i,]$value, probs=0.975)
  luxembourg_median_pis[nrow(luxembourg_median_pis)+1,] <- c(i, luxembourg_median, luxembourg_lower, luxembourg_upper)
}

#netherlands
netherlands_xarima <- auto.arima(y=netherlands$ppw, max.p=5, max.q=5,
                                 xreg=as.matrix(netherlands[, c('GDP', 'Population', 'co2', 'exports')]))

# refit on pre-2018 data to compute genuine test set residuals
netherlands_train <- netherlands[netherlands$Year < 2018, ]
netherlands_test  <- netherlands[netherlands$Year >= 2018, ]

netherlands_xarima_test <- auto.arima(y=netherlands_train$ppw, max.p=5, max.q=5,
                                      xreg=as.matrix(netherlands_train[, c('GDP', 'Population', 'co2', 'exports')]))

netherlands_test_preds <- as.numeric(predict(netherlands_xarima_test,
                                             newxreg=as.matrix(netherlands_test[, c('GDP', 'Population', 'co2', 'exports')]),
                                             n.ahead=nrow(netherlands_test))$pred)
netherlands_test_residuals <- netherlands_test$ppw - netherlands_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
netherlands_centred_residuals <- netherlands_test_residuals - mean(netherlands_test_residuals)

# blank dataframe to store the results
netherlands_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

# iterate over the sim ids
set.seed(123)
for (i in 1:1000){
  netherlands_newdata <- data.frame(
    GDP=gdp_forecasts[gdp_forecasts$country=='Netherlands'&gdp_forecasts$sim_id==i,'value'],
    Population=population_forecasts[population_forecasts$country=='Netherlands'&population_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Netherlands'&co2_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Netherlands'&exports_forecasts$sim_id==i,'value']
  )
  netherlands_forecast <- as.numeric(predict(netherlands_xarima,
                                             newxreg=as.matrix(netherlands_newdata[, c('GDP', 'Population', 'co2', 'exports')]),
                                             n.ahead=8)$pred)
  
  netherlands_boot_residual <- sample(netherlands_centred_residuals, size=1)
  netherlands_forecast_with_residuals <- netherlands_forecast + netherlands_boot_residual
  
  netherlands_temp <- data.frame(year=2023:2030, sim_id=i,
                                 value=netherlands_forecast_with_residuals)
  netherlands_plastic_forecasts <- rbind(netherlands_plastic_forecasts, netherlands_temp)
}

# blank dataframe for median forecast and PIs
netherlands_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                     lower_pi=numeric(), upper_pi=numeric())

for(i in 2023:2030){
  netherlands_median <- median(netherlands_plastic_forecasts[netherlands_plastic_forecasts$year==i,]$value)
  netherlands_lower  <- quantile(netherlands_plastic_forecasts[netherlands_plastic_forecasts$year==i,]$value, probs=0.025)
  netherlands_upper  <- quantile(netherlands_plastic_forecasts[netherlands_plastic_forecasts$year==i,]$value, probs=0.975)
  netherlands_median_pis[nrow(netherlands_median_pis)+1,] <- c(i, netherlands_median, netherlands_lower, netherlands_upper)
}

#portugal
portugal_bayes_ridge <- bayesglm(ppw~material_footprint, data=portugal,
                                 family=gaussian)

# refit on pre-2018 data to compute genuine test set residuals
portugal_train <- portugal[portugal$Year < 2018, ]
portugal_test  <- portugal[portugal$Year >= 2018, ]

portugal_bayes_ridge_test <- bayesglm(ppw~material_footprint, data=portugal_train,
                                      family=gaussian)

portugal_test_preds <- predict(portugal_bayes_ridge_test,
                               newdata=portugal_test[, c('material_footprint'), drop=FALSE])
portugal_test_residuals <- portugal_test$ppw - portugal_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
portugal_centred_residuals <- portugal_test_residuals - mean(portugal_test_residuals)

# blank dataframe to store the results
portugal_plastic_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                         sim_id=rep(1:1000, each=8), value=NA)

set.seed(123)
for (i in 1:1000){
  portugal_newdata <- data.frame(
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Portugal'&material_footprint_forecasts$sim_id==i,'value']
  )
  portugal_forecast <- predict(portugal_bayes_ridge, newdata=portugal_newdata)
  
  portugal_boot_residual <- sample(portugal_centred_residuals, size=1)
  portugal_plastic_forecasts$value[portugal_plastic_forecasts$sim_id==i] <- portugal_forecast + portugal_boot_residual
}

# blank dataframe for median forecast and PIs
portugal_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                  lower_pi=numeric(), upper_pi=numeric())

for(i in 2023:2030){
  portugal_median <- median(portugal_plastic_forecasts[portugal_plastic_forecasts$year==i,]$value)
  portugal_lower  <- quantile(portugal_plastic_forecasts[portugal_plastic_forecasts$year==i,]$value, probs=0.025)
  portugal_upper  <- quantile(portugal_plastic_forecasts[portugal_plastic_forecasts$year==i,]$value, probs=0.975)
  portugal_median_pis[nrow(portugal_median_pis)+1,] <- c(i, portugal_median, portugal_lower, portugal_upper)
}

#spain
spain_xarima <- auto.arima(y=spain$ppw, max.p=5, max.q=5,
                           xreg=as.matrix(spain[, c('Population', 'GDP', 'exports')]))

# refit on pre-2018 data to compute genuine test set residuals
spain_train <- spain[spain$Year < 2018, ]
spain_test  <- spain[spain$Year >= 2018, ]

spain_xarima_test <- auto.arima(y=spain_train$ppw, max.p=5, max.q=5,
                                xreg=as.matrix(spain_train[, c('Population', 'GDP', 'exports')]))

spain_test_preds <- as.numeric(predict(spain_xarima_test,
                                       newxreg=as.matrix(spain_test[, c('Population', 'GDP', 'exports')]),
                                       n.ahead=nrow(spain_test))$pred)
spain_test_residuals <- spain_test$ppw - spain_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
spain_centred_residuals <- spain_test_residuals - mean(spain_test_residuals)

# blank dataframe to store the results
spain_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for (i in 1:1000){
  spain_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Spain'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Spain'&gdp_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Spain'&exports_forecasts$sim_id==i,'value']
  )
  spain_forecast <- as.numeric(predict(spain_xarima,
                                       newxreg=as.matrix(spain_newdata[, c('Population', 'GDP', 'exports')]),
                                       n.ahead=8)$pred)
  
  spain_boot_residual <- sample(spain_centred_residuals, size=1)
  spain_forecast_with_residuals <- spain_forecast + spain_boot_residual
  
  spain_temp <- data.frame(year=2023:2030, sim_id=i,
                           value=spain_forecast_with_residuals)
  spain_plastic_forecasts <- rbind(spain_plastic_forecasts, spain_temp)
}

# blank dataframe for median forecast and PIs
spain_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                               lower_pi=numeric(), upper_pi=numeric())

for(i in 2023:2030){
  spain_median <- median(spain_plastic_forecasts[spain_plastic_forecasts$year==i,]$value)
  spain_lower  <- quantile(spain_plastic_forecasts[spain_plastic_forecasts$year==i,]$value, probs=0.025)
  spain_upper  <- quantile(spain_plastic_forecasts[spain_plastic_forecasts$year==i,]$value, probs=0.975)
  spain_median_pis[nrow(spain_median_pis)+1,] <- c(i, spain_median, spain_lower, spain_upper)
}

#sweden
sweden_svm <- svm(ppw~GDP+co2, data=sweden, kernel='radial',
                  cost=5, gamma=0.5, epsilon=0.091)

# refit on pre-2018 data to compute genuine test set residuals
sweden_train <- sweden[sweden$Year < 2018, ]
sweden_test  <- sweden[sweden$Year >= 2018, ]

sweden_svm_test <- svm(ppw~GDP+co2, data=sweden_train, kernel='radial',
                       cost=5, gamma=0.5, epsilon=0.091)

sweden_test_preds <- predict(sweden_svm_test,
                             newdata=sweden_test[, c('GDP', 'co2')])
sweden_test_residuals <- sweden_test$ppw - sweden_test_preds

# centre residuals by subtracting their mean (Carpenter & Bithell, 2000, Section 2.3)
sweden_centred_residuals <- sweden_test_residuals - mean(sweden_test_residuals)

# blank dataframe to store the results
sweden_plastic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for (i in 1:1000){
  sweden_newdata <- data.frame(
    GDP=gdp_forecasts[gdp_forecasts$country=='Sweden'&gdp_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Sweden'&co2_forecasts$sim_id==i,'value']
  )
  sweden_forecast <- predict(sweden_svm, newdata=sweden_newdata)
  
  sweden_boot_residual <- sample(sweden_centred_residuals, size=1)
  sweden_forecast_with_residuals <- sweden_forecast + sweden_boot_residual
  
  sweden_temp <- data.frame(year=2023:2030, sim_id=i,
                            value=sweden_forecast_with_residuals)
  sweden_plastic_forecasts <- rbind(sweden_plastic_forecasts, sweden_temp)
}

# blank dataframe for median forecast and PIs
sweden_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                lower_pi=numeric(), upper_pi=numeric())

for(i in 2023:2030){
  sweden_median <- median(sweden_plastic_forecasts[sweden_plastic_forecasts$year==i,]$value)
  sweden_lower  <- quantile(sweden_plastic_forecasts[sweden_plastic_forecasts$year==i,]$value, probs=0.025)
  sweden_upper  <- quantile(sweden_plastic_forecasts[sweden_plastic_forecasts$year==i,]$value, probs=0.975)
  sweden_median_pis[nrow(sweden_median_pis)+1,] <- c(i, sweden_median, sweden_lower, sweden_upper)
}


#set up the dataframe----
plastic_meta <- data.frame(country=c("Austria", "Belgium", "Denmark", "Finland", 
                                   "France", "Germany","Ireland", "Italy", "Luxembourg", 
                                   "Netherlands", "Portugal", "Spain", "Sweden"),
                         stream="plastic",
                         model=c('SVM', 'XARIMA', 'SVM', 'BRR', 'SVM', 'SVM', 'SVM',
                                 'BRR', 'XARIMA', 'XARIMA', 'BRR', 'XARIMA', 'SVM'),
                         n_covariates=c(3, 2, 2, 2, 4, 3, 6, 4, 5, 4, 1, 3, 2)
)

#2023 forcast error & absolute forecast error----
plastic_meta$forecast_error_2023 <- c(plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Austria-austria_median_pis[austria_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Belgium-belgium_median_pis[belgium_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Denmark-denmark_median_pis[denmark_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Finland-finland_median_pis[finland_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$France-france_median_pis[france_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Germany-germany_median_pis[germany_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Ireland-ireland_median_pis[ireland_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Italy-italy_median_pis[italy_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Luxembourg-luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Netherlands-netherlands_median_pis[netherlands_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Portugal-portugal_median_pis[portugal_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Spain-spain_median_pis[spain_median_pis$year=='2023',]$median_forecast,
                                    plastic_packaging_waste[plastic_packaging_waste$TIME=='2023',]$Sweden-sweden_median_pis[sweden_median_pis$year=='2023',]$median_forecast
)

plastic_meta$abs_forecast_error_2023 <- abs(plastic_meta$forecast_error_2023)

#PI width----
plastic_meta$pi_width_2023 <- c(austria_median_pis[austria_median_pis$year=='2023',]$upper_pi-austria_median_pis[austria_median_pis$year=='2023',]$lower_pi,
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
plastic_meta$train_sd <- c(sd(austria[austria$Year<=2017,]$ppw),
                         sd(belgium[belgium$Year<=2017,]$ppw),
                         sd(denmark[denmark$Year<=2017,]$ppw),
                         sd(finland[finland$Year<=2017,]$ppw),
                         sd(france[france$Year<=2017,]$ppw),
                         sd(germany[germany$Year<=2017,]$ppw),
                         sd(ireland[ireland$Year<=2017,]$ppw),
                         sd(italy[italy$Year<=2017,]$ppw),
                         sd(luxembourg[luxembourg$Year<=2017,]$ppw),
                         sd(netherlands[netherlands$Year<=2017,]$ppw),
                         sd(portugal[portugal$Year<=2017,]$ppw),
                         sd(spain[spain$Year<=2017,]$ppw),
                         sd(sweden[sweden$Year<=2017,]$ppw)
)

plastic_meta$test_sd <- c(sd(austria[austria$Year>=2018,]$ppw),
                        sd(belgium[belgium$Year>=2018,]$ppw),
                        sd(denmark[denmark$Year>=2018,]$ppw),
                        sd(finland[finland$Year>=2018,]$ppw),
                        sd(france[france$Year>=2018,]$ppw),
                        sd(germany[germany$Year>=2018,]$ppw),
                        sd(ireland[ireland$Year>=2018,]$ppw),
                        sd(italy[italy$Year>=2018,]$ppw),
                        sd(luxembourg[luxembourg$Year>=2018,]$ppw),
                        sd(netherlands[netherlands$Year>=2018,]$ppw),
                        sd(portugal[portugal$Year>=2018,]$ppw),
                        sd(spain[spain$Year>=2018,]$ppw),
                        sd(sweden[sweden$Year>=2018,]$ppw)
)

plastic_meta$sd_ratio <- plastic_meta$test_sd/plastic_meta$train_sd

#trend of training set, trend of whole series, their difference, binary change of trend direction----
trend_slope <- function(df, year_col = "Year", y_col = "ppw"){
  coef(lm(df[[y_col]] ~ df[[year_col]]))[2]
}

plastic_meta$slope_full <- c(trend_slope(austria), trend_slope(belgium),
                           trend_slope(denmark), trend_slope(finland),
                           trend_slope(france), trend_slope(germany),
                           trend_slope(ireland), trend_slope(italy),
                           trend_slope(luxembourg), trend_slope(netherlands),
                           trend_slope(portugal), trend_slope(spain),
                           trend_slope(sweden)
)

plastic_meta$slope_train <- c(
  trend_slope(austria[austria$Year<=2017,]), trend_slope(belgium[belgium$Year<=2017,]),
  trend_slope(denmark[denmark$Year<=2017,]), trend_slope(finland[finland$Year<=2017,]),
  trend_slope(france[france$Year<=2017,]), trend_slope(germany[germany$Year<=2017,]),
  trend_slope(ireland[ireland$Year<=2017,]), trend_slope(italy[italy$Year<=2017,]),
  trend_slope(luxembourg[luxembourg$Year<=2017,]), trend_slope(netherlands[netherlands$Year<=2017,]),
  trend_slope(portugal[portugal$Year<=2017,]), trend_slope(spain[spain$Year<=2017,]),
  trend_slope(sweden[sweden$Year<=2017,])
)

plastic_meta$slope_diff <- plastic_meta$slope_full - plastic_meta$slope_train


plastic_meta$trend_reversal <- as.integer(sign(plastic_meta$slope_full)!=sign(plastic_meta$slope_train))

#maximum and mean absolute change (Rise/fall) over the whole series----
max_abs_change <- function(df, y_col = "ppw"){
  max(abs(diff(df[[y_col]])))
}

mean_abs_change <- function(df, y_col = "ppw"){
  mean(abs(diff(df[[y_col]])))
}

plastic_meta$max_abs_change <- c(max_abs_change(austria), max_abs_change(belgium),
                               max_abs_change(denmark), max_abs_change(finland),
                               max_abs_change(france), max_abs_change(germany),
                               max_abs_change(ireland), max_abs_change(italy),
                               max_abs_change(luxembourg), max_abs_change(netherlands),
                               max_abs_change(portugal), max_abs_change(spain),
                               max_abs_change(sweden)
)

plastic_meta$mean_abs_change <- c(mean_abs_change(austria), mean_abs_change(belgium),
                                mean_abs_change(denmark), mean_abs_change(finland),
                                mean_abs_change(france), mean_abs_change(germany),
                                mean_abs_change(ireland), mean_abs_change(italy), 
                                mean_abs_change(luxembourg), mean_abs_change(netherlands),
                                mean_abs_change(portugal), mean_abs_change(spain),
                                mean_abs_change(sweden)
)

#writing the csv file----
write.csv(plastic_meta, file='plastic_meta_dataset.csv')

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
metallic_packaging_waste <- read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Packaging waste forecasting/Packaging Data/metallic_packaging_waste.xlsx")
metallic_packaging_waste$TIME <- as.double(metallic_packaging_waste$TIME)

#training data
countries <- c("Austria", "Belgium", "Denmark", "Finland", "France", "Germany",
               "Ireland", "Italy", "Luxembourg", "Netherlands", "Portugal", "Spain", "Sweden")
vars <- c("Year", "Population", "GDP", "material_footprint", "energy_consumption", "co2", "exports")

modelling_data <- lapply(setNames(countries, tolower(countries)), function(ctry) {
  df <- read_excel(paste0("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Packaging waste forecasting/Country data/", ctry, ".xlsx")) |>
    dplyr::select(all_of(vars)) |>
    filter(Year >= 1997, Year <= 2022)
  df$mpw <- metallic_packaging_waste[[ctry]][metallic_packaging_waste$TIME <= 2022]
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

#austria----
set.seed(123)
austria_svm <- svm(mpw~Population+material_footprint, data=austria, kernel='radial',
                   cost=2.7, gamma=0.05, epsilon=0.041)

austria_train <- austria[austria$Year < 2018, ]
austria_test  <- austria[austria$Year >= 2018, ]

set.seed(123)
austria_svm_test <- svm(mpw~Population+material_footprint, data=austria_train, kernel='radial',
                        cost=2.7, gamma=0.05, epsilon=0.041)

austria_test_preds <- predict(austria_svm_test,
                              newdata=austria_test[, c('Population', 'material_footprint')])
austria_test_residuals <- austria_test$mpw - austria_test_preds
austria_centred_residuals <- austria_test_residuals - mean(austria_test_residuals)

austria_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  austria_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Austria'&population_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Austria'&material_footprint_forecasts$sim_id==i,'value']
  )
  austria_forecast <- predict(austria_svm, newdata=austria_newdata)
  austria_boot_residual <- sample(austria_centred_residuals, size=1)
  austria_temp <- data.frame(year=2023:2030, sim_id=i, value=austria_forecast + austria_boot_residual)
  austria_metallic_forecasts <- rbind(austria_metallic_forecasts, austria_temp)
}

austria_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  austria_median <- median(austria_metallic_forecasts[austria_metallic_forecasts$year==i,]$value)
  austria_lower  <- quantile(austria_metallic_forecasts[austria_metallic_forecasts$year==i,]$value, probs=0.025)
  austria_upper  <- quantile(austria_metallic_forecasts[austria_metallic_forecasts$year==i,]$value, probs=0.975)
  austria_median_pis[nrow(austria_median_pis)+1,] <- c(i, austria_median, austria_lower, austria_upper)
}

plot(y=austria$mpw, x=austria$Year, type='b',
     xlab='Year', ylab='%', main='Austria Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(austria$mpw), max(austria_median_pis$upper_pi)))
lines(y=austria_median_pis$median_forecast, x=austria_median_pis$year, type='b', col='blue')
polygon(c(austria_median_pis$year, rev(austria_median_pis$year)),
        c(austria_median_pis$upper_pi, rev(austria_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#belgium----
belgium_xarima <- auto.arima(y=belgium$mpw, max.p=5, max.q=5,
                             xreg=as.matrix(belgium[, c('GDP')]))

belgium_train <- belgium[belgium$Year < 2018, ]
belgium_test  <- belgium[belgium$Year >= 2018, ]

belgium_xarima_test <- auto.arima(y=belgium_train$mpw, max.p=5, max.q=5,
                                  xreg=as.matrix(belgium_train[, c('GDP')]))

belgium_test_preds <- as.numeric(predict(belgium_xarima_test,
                                         newxreg=as.matrix(belgium_test[, c('GDP')]),
                                         n.ahead=nrow(belgium_test))$pred)
belgium_test_residuals <- belgium_test$mpw - belgium_test_preds
belgium_centred_residuals <- belgium_test_residuals - mean(belgium_test_residuals)

belgium_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  belgium_newdata <- data.frame(GDP=gdp_forecasts[gdp_forecasts$country=='Belgium'&gdp_forecasts$sim_id==i,'value'])
  belgium_forecast <- as.numeric(predict(belgium_xarima,
                                         newxreg=as.matrix(belgium_newdata[, c('GDP')]),
                                         n.ahead=8)$pred)
  belgium_boot_residual <- sample(belgium_centred_residuals, size=1)
  belgium_forecast_with_residuals <- belgium_forecast + belgium_boot_residual
  belgium_temp <- data.frame(year=2023:2030, sim_id=i, value=belgium_forecast_with_residuals)
  belgium_metallic_forecasts <- rbind(belgium_metallic_forecasts, belgium_temp)
}

belgium_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  belgium_median <- median(belgium_metallic_forecasts[belgium_metallic_forecasts$year==i,]$value)
  belgium_lower  <- quantile(belgium_metallic_forecasts[belgium_metallic_forecasts$year==i,]$value, probs=0.025)
  belgium_upper  <- quantile(belgium_metallic_forecasts[belgium_metallic_forecasts$year==i,]$value, probs=0.975)
  belgium_median_pis[nrow(belgium_median_pis)+1,] <- c(i, belgium_median, belgium_lower, belgium_upper)
}

plot(y=belgium$mpw, x=belgium$Year, type='b',
     xlab='Year', ylab='%', main='Belgium Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(belgium$mpw), max(belgium_median_pis$upper_pi)))
lines(y=belgium_median_pis$median_forecast, x=belgium_median_pis$year, type='b', col='blue')
polygon(c(belgium_median_pis$year, rev(belgium_median_pis$year)),
        c(belgium_median_pis$upper_pi, rev(belgium_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#denmark----
set.seed(20229798)
ss <- AddDynamicRegression(list(), mpw~GDP, data=denmark)
denmark_bsts <- bsts(mpw~GDP, state.specification=ss, data=denmark, niter=1000)

denmark_train <- denmark[denmark$Year < 2018, ]
denmark_test  <- denmark[denmark$Year >= 2018, ]

set.seed(20229798)
ss_test <- AddDynamicRegression(list(), mpw~GDP, data=denmark_train)
denmark_bsts_test <- bsts(mpw~GDP, state.specification=ss_test, data=denmark_train, niter=1000)

denmark_test_preds <- colMeans(predict(denmark_bsts_test,
                                       newdata=denmark_test[, 'GDP', drop=FALSE],
                                       burn=100)$distribution)
denmark_test_residuals <- denmark_test$mpw - denmark_test_preds
denmark_centred_residuals <- denmark_test_residuals - mean(denmark_test_residuals)

denmark_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(20229798)
for(i in 1:1000){
  denmark_newdata <- data.frame(GDP=gdp_forecasts[gdp_forecasts$country=='Denmark'&gdp_forecasts$sim_id==i,'value'])
  denmark_forecast <- predict(denmark_bsts, h=8, newdata=denmark_newdata, burn=100)$mean
  denmark_boot_residual <- sample(denmark_centred_residuals, size=1)
  denmark_forecast_with_residuals <- denmark_forecast + denmark_boot_residual
  denmark_temp <- data.frame(year=2023:2030, sim_id=i, value=denmark_forecast_with_residuals)
  denmark_metallic_forecasts <- rbind(denmark_metallic_forecasts, denmark_temp)
}

denmark_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  denmark_median <- median(denmark_metallic_forecasts[denmark_metallic_forecasts$year==i,]$value)
  denmark_lower  <- quantile(denmark_metallic_forecasts[denmark_metallic_forecasts$year==i,]$value, probs=0.025)
  denmark_upper  <- quantile(denmark_metallic_forecasts[denmark_metallic_forecasts$year==i,]$value, probs=0.975)
  denmark_median_pis[nrow(denmark_median_pis)+1,] <- c(i, denmark_median, denmark_lower, denmark_upper)
}

plot(y=denmark$mpw, x=denmark$Year, type='b',
     xlab='Year', ylab='%', main='Denmark Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(denmark$mpw), max(denmark_median_pis$upper_pi)))
lines(y=denmark_median_pis$median_forecast, x=denmark_median_pis$year, type='b', col='blue')
polygon(c(denmark_median_pis$year, rev(denmark_median_pis$year)),
        c(denmark_median_pis$upper_pi, rev(denmark_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#finland----
finland_svm <- svm(mpw~Population+GDP, data=finland, kernel='radial',
                   cost=1.1, gamma=0.05, epsilon=0.001)

finland_train <- finland[finland$Year < 2018, ]
finland_test  <- finland[finland$Year >= 2018, ]

finland_svm_test <- svm(mpw~Population+GDP, data=finland_train, kernel='radial',
                        cost=1.1, gamma=0.05, epsilon=0.001)

finland_test_preds <- predict(finland_svm_test,
                              newdata=finland_test[, c('Population', 'GDP')])
finland_test_residuals <- finland_test$mpw - finland_test_preds
finland_centred_residuals <- finland_test_residuals - mean(finland_test_residuals)

finland_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  finland_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Finland'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Finland'&gdp_forecasts$sim_id==i,'value']
  )
  finland_forecast <- predict(finland_svm, newdata=finland_newdata)
  finland_boot_residual <- sample(finland_centred_residuals, size=1)
  finland_temp <- data.frame(year=2023:2030, sim_id=i, value=finland_forecast + finland_boot_residual)
  finland_metallic_forecasts <- rbind(finland_metallic_forecasts, finland_temp)
}

finland_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  finland_median <- median(finland_metallic_forecasts[finland_metallic_forecasts$year==i,]$value)
  finland_lower  <- quantile(finland_metallic_forecasts[finland_metallic_forecasts$year==i,]$value, probs=0.025)
  finland_upper  <- quantile(finland_metallic_forecasts[finland_metallic_forecasts$year==i,]$value, probs=0.975)
  finland_median_pis[nrow(finland_median_pis)+1,] <- c(i, finland_median, finland_lower, finland_upper)
}

plot(y=finland$mpw, x=finland$Year, type='b',
     xlab='Year', ylab='%', main='Finland Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(finland$mpw), max(finland_median_pis$upper_pi)))
lines(y=finland_median_pis$median_forecast, x=finland_median_pis$year, type='b', col='blue')
polygon(c(finland_median_pis$year, rev(finland_median_pis$year)),
        c(finland_median_pis$upper_pi, rev(finland_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#france----
set.seed(123)
france_svm <- svm(mpw~Population+co2+exports+material_footprint, data=france, kernel='radial',
                  cost=1.8, gamma=0.05, epsilon=0.001)

france_train <- france[france$Year < 2018, ]
france_test  <- france[france$Year >= 2018, ]

set.seed(123)
france_svm_test <- svm(mpw~Population+co2+exports+material_footprint, data=france_train, kernel='radial',
                       cost=1.8, gamma=0.05, epsilon=0.001)

france_test_preds <- predict(france_svm_test,
                             newdata=france_test[, c('Population', 'co2', 'exports', 'material_footprint')])
france_test_residuals <- france_test$mpw - france_test_preds
france_centred_residuals <- france_test_residuals - mean(france_test_residuals)

france_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  france_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='France'&population_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='France'&co2_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='France'&exports_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='France'&material_footprint_forecasts$sim_id==i,'value']
  )
  france_forecast <- predict(france_svm, newdata=france_newdata)
  france_boot_residual <- sample(france_centred_residuals, size=1)
  france_temp <- data.frame(year=2023:2030, sim_id=i, value=france_forecast + france_boot_residual)
  france_metallic_forecasts <- rbind(france_metallic_forecasts, france_temp)
}

france_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  france_median <- median(france_metallic_forecasts[france_metallic_forecasts$year==i,]$value)
  france_lower  <- quantile(france_metallic_forecasts[france_metallic_forecasts$year==i,]$value, probs=0.025)
  france_upper  <- quantile(france_metallic_forecasts[france_metallic_forecasts$year==i,]$value, probs=0.975)
  france_median_pis[nrow(france_median_pis)+1,] <- c(i, france_median, france_lower, france_upper)
}

plot(y=france$mpw, x=france$Year, type='b',
     xlab='Year', ylab='%', main='France Metallic Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=france_median_pis$median_forecast, x=france_median_pis$year, type='b', col='blue')
polygon(c(france_median_pis$year, rev(france_median_pis$year)),
        c(france_median_pis$upper_pi, rev(france_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#germany----
germany_xarima <- auto.arima(y=germany$mpw, max.p=5, max.q=5,
                             xreg=as.matrix(germany[, c('co2', 'exports')]))

germany_train <- germany[germany$Year < 2018, ]
germany_test  <- germany[germany$Year >= 2018, ]

germany_xarima_test <- auto.arima(y=germany_train$mpw, max.p=5, max.q=5,
                                  xreg=as.matrix(germany_train[, c('co2', 'exports')]))

germany_test_preds <- as.numeric(predict(germany_xarima_test,
                                         newxreg=as.matrix(germany_test[, c('co2', 'exports')]),
                                         n.ahead=nrow(germany_test))$pred)
germany_test_residuals <- germany_test$mpw - germany_test_preds
germany_centred_residuals <- germany_test_residuals - mean(germany_test_residuals)

germany_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  germany_newdata <- data.frame(
    co2=co2_forecasts[co2_forecasts$country=='Germany'&co2_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Germany'&exports_forecasts$sim_id==i,'value']
  )
  germany_forecast <- as.numeric(predict(germany_xarima,
                                         newxreg=as.matrix(germany_newdata[, c('co2', 'exports')]),
                                         n.ahead=8)$pred)
  germany_boot_residual <- sample(germany_centred_residuals, size=1)
  germany_forecast_with_residuals <- germany_forecast + germany_boot_residual
  germany_temp <- data.frame(year=2023:2030, sim_id=i, value=germany_forecast_with_residuals)
  germany_metallic_forecasts <- rbind(germany_metallic_forecasts, germany_temp)
}

germany_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  germany_median <- median(germany_metallic_forecasts[germany_metallic_forecasts$year==i,]$value)
  germany_lower  <- quantile(germany_metallic_forecasts[germany_metallic_forecasts$year==i,]$value, probs=0.025)
  germany_upper  <- quantile(germany_metallic_forecasts[germany_metallic_forecasts$year==i,]$value, probs=0.975)
  germany_median_pis[nrow(germany_median_pis)+1,] <- c(i, germany_median, germany_lower, germany_upper)
}

plot(y=germany$mpw, x=germany$Year, type='b',
     xlab='Year', ylab='%', main='Germany Metallic Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=germany_median_pis$median_forecast, x=germany_median_pis$year, type='b', col='blue')
polygon(c(germany_median_pis$year, rev(germany_median_pis$year)),
        c(germany_median_pis$upper_pi, rev(germany_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#ireland----
set.seed(123)
ireland_svm <- svm(mpw~Population+GDP, data=ireland, kernel='radial',
                   cost=5, gamma=0.05, epsilon=0.061)

ireland_train <- ireland[ireland$Year < 2018, ]
ireland_test  <- ireland[ireland$Year >= 2018, ]

set.seed(123)
ireland_svm_test <- svm(mpw~Population+GDP, data=ireland_train, kernel='radial',
                        cost=5, gamma=0.05, epsilon=0.061)

ireland_test_preds <- predict(ireland_svm_test,
                              newdata=ireland_test[, c('Population', 'GDP')])
ireland_test_residuals <- ireland_test$mpw - ireland_test_preds
ireland_centred_residuals <- ireland_test_residuals - mean(ireland_test_residuals)

ireland_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  ireland_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Ireland'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Ireland'&gdp_forecasts$sim_id==i,'value']
  )
  ireland_forecast <- predict(ireland_svm, newdata=ireland_newdata)
  ireland_boot_residual <- sample(ireland_centred_residuals, size=1)
  ireland_temp <- data.frame(year=2023:2030, sim_id=i, value=ireland_forecast + ireland_boot_residual)
  ireland_metallic_forecasts <- rbind(ireland_metallic_forecasts, ireland_temp)
}

ireland_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  ireland_median <- median(ireland_metallic_forecasts[ireland_metallic_forecasts$year==i,]$value)
  ireland_lower  <- quantile(ireland_metallic_forecasts[ireland_metallic_forecasts$year==i,]$value, probs=0.025)
  ireland_upper  <- quantile(ireland_metallic_forecasts[ireland_metallic_forecasts$year==i,]$value, probs=0.975)
  ireland_median_pis[nrow(ireland_median_pis)+1,] <- c(i, ireland_median, ireland_lower, ireland_upper)
}

plot(y=ireland$mpw, x=ireland$Year, type='b',
     xlab='Year', ylab='%', main='Ireland Metallic Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=ireland_median_pis$median_forecast, x=ireland_median_pis$year, type='b', col='blue')
polygon(c(ireland_median_pis$year, rev(ireland_median_pis$year)),
        c(ireland_median_pis$upper_pi, rev(ireland_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#italy----
italy_cubist <- cubist(x=as.matrix(italy[, c('Population', 'GDP', 'energy_consumption')]),
                       y=italy$mpw, committees=5)

italy_train <- italy[italy$Year < 2018, ]
italy_test  <- italy[italy$Year >= 2018, ]

italy_cubist_test <- cubist(x=as.matrix(italy_train[, c('Population', 'GDP', 'energy_consumption')]),
                            y=italy_train$mpw, committees=5)

italy_test_preds <- predict(italy_cubist_test,
                            newdata=italy_test[, c('Population', 'GDP', 'energy_consumption')],
                            neighbors=7)
italy_test_residuals <- italy_test$mpw - italy_test_preds
italy_centred_residuals <- italy_test_residuals - mean(italy_test_residuals)

italy_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  italy_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Italy'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Italy'&gdp_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Italy'&energy_consumption_forecasts$sim_id==i,'value']
  )
  italy_forecast <- predict(italy_cubist, newdata=italy_newdata, neighbors=7)
  italy_boot_residual <- sample(italy_centred_residuals, size=1)
  italy_temp <- data.frame(year=2023:2030, sim_id=i, value=italy_forecast + italy_boot_residual)
  italy_metallic_forecasts <- rbind(italy_metallic_forecasts, italy_temp)
}

italy_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                               lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  italy_median <- median(italy_metallic_forecasts[italy_metallic_forecasts$year==i,]$value)
  italy_lower  <- quantile(italy_metallic_forecasts[italy_metallic_forecasts$year==i,]$value, probs=0.025)
  italy_upper  <- quantile(italy_metallic_forecasts[italy_metallic_forecasts$year==i,]$value, probs=0.975)
  italy_median_pis[nrow(italy_median_pis)+1,] <- c(i, italy_median, italy_lower, italy_upper)
}

plot(y=italy$mpw, x=italy$Year, type='b',
     xlab='Year', ylab='%', main='Italy Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(italy$mpw), max(italy_median_pis$upper_pi)))
lines(y=italy_median_pis$median_forecast, x=italy_median_pis$year, type='b', col='blue')
polygon(c(italy_median_pis$year, rev(italy_median_pis$year)),
        c(italy_median_pis$upper_pi, rev(italy_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#luxembourg----
luxembourg_svm <- svm(mpw~GDP+energy_consumption, data=luxembourg, kernel='radial',
                      cost=1.4, gamma=0.05, epsilon=0.071)

luxembourg_train <- luxembourg[luxembourg$Year < 2018, ]
luxembourg_test  <- luxembourg[luxembourg$Year >= 2018, ]

luxembourg_svm_test <- svm(mpw~GDP+energy_consumption, data=luxembourg_train, kernel='radial',
                           cost=1.4, gamma=0.05, epsilon=0.071)

luxembourg_test_preds <- predict(luxembourg_svm_test,
                                 newdata=luxembourg_test[, c('GDP', 'energy_consumption')])
luxembourg_test_residuals <- luxembourg_test$mpw - luxembourg_test_preds
luxembourg_centred_residuals <- luxembourg_test_residuals - mean(luxembourg_test_residuals)

luxembourg_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  luxembourg_newdata <- data.frame(
    GDP=gdp_forecasts[gdp_forecasts$country=='Luxembourg'&gdp_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Luxembourg'&energy_consumption_forecasts$sim_id==i,'value']
  )
  luxembourg_forecast <- predict(luxembourg_svm, newdata=luxembourg_newdata)
  luxembourg_boot_residual <- sample(luxembourg_centred_residuals, size=1)
  luxembourg_temp <- data.frame(year=2023:2030, sim_id=i, value=luxembourg_forecast + luxembourg_boot_residual)
  luxembourg_metallic_forecasts <- rbind(luxembourg_metallic_forecasts, luxembourg_temp)
}

luxembourg_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                    lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  luxembourg_median <- median(luxembourg_metallic_forecasts[luxembourg_metallic_forecasts$year==i,]$value)
  luxembourg_lower  <- quantile(luxembourg_metallic_forecasts[luxembourg_metallic_forecasts$year==i,]$value, probs=0.025)
  luxembourg_upper  <- quantile(luxembourg_metallic_forecasts[luxembourg_metallic_forecasts$year==i,]$value, probs=0.975)
  luxembourg_median_pis[nrow(luxembourg_median_pis)+1,] <- c(i, luxembourg_median, luxembourg_lower, luxembourg_upper)
}

plot(y=luxembourg$mpw, x=luxembourg$Year, type='b',
     xlab='Year', ylab='%', main='Luxembourg Metallic Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=luxembourg_median_pis$median_forecast, x=luxembourg_median_pis$year, type='b', col='blue')
polygon(c(luxembourg_median_pis$year, rev(luxembourg_median_pis$year)),
        c(luxembourg_median_pis$upper_pi, rev(luxembourg_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#netherlands----
set.seed(123)
netherlands_svm <- svm(mpw~Population, data=netherlands, kernel='radial',
                       cost=5, gamma=0.15, epsilon=0.081)

netherlands_train <- netherlands[netherlands$Year < 2018, ]
netherlands_test  <- netherlands[netherlands$Year >= 2018, ]

set.seed(123)
netherlands_svm_test <- svm(mpw~Population, data=netherlands_train, kernel='radial',
                            cost=5, gamma=0.15, epsilon=0.081)

netherlands_test_preds <- predict(netherlands_svm_test,
                                  newdata=netherlands_test[, 'Population', drop=FALSE])
netherlands_test_residuals <- netherlands_test$mpw - netherlands_test_preds
netherlands_centred_residuals <- netherlands_test_residuals - mean(netherlands_test_residuals)

netherlands_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  netherlands_newdata <- data.frame(Population=population_forecasts[population_forecasts$country=='Netherlands'&population_forecasts$sim_id==i,'value'])
  netherlands_forecast <- predict(netherlands_svm, newdata=netherlands_newdata)
  netherlands_boot_residual <- sample(netherlands_centred_residuals, size=1)
  netherlands_temp <- data.frame(year=2023:2030, sim_id=i, value=netherlands_forecast + netherlands_boot_residual)
  netherlands_metallic_forecasts <- rbind(netherlands_metallic_forecasts, netherlands_temp)
}

netherlands_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                     lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  netherlands_median <- median(netherlands_metallic_forecasts[netherlands_metallic_forecasts$year==i,]$value)
  netherlands_lower  <- quantile(netherlands_metallic_forecasts[netherlands_metallic_forecasts$year==i,]$value, probs=0.025)
  netherlands_upper  <- quantile(netherlands_metallic_forecasts[netherlands_metallic_forecasts$year==i,]$value, probs=0.975)
  netherlands_median_pis[nrow(netherlands_median_pis)+1,] <- c(i, netherlands_median, netherlands_lower, netherlands_upper)
}

plot(y=netherlands$mpw, x=netherlands$Year, type='b',
     xlab='Year', ylab='%', main='Netherlands Metallic Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=netherlands_median_pis$median_forecast, x=netherlands_median_pis$year, type='b', col='blue')
polygon(c(netherlands_median_pis$year, rev(netherlands_median_pis$year)),
        c(netherlands_median_pis$upper_pi, rev(netherlands_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#portugal----
portugal_cubist <- cubist(x=as.matrix(portugal[, c('Population', 'material_footprint')]),
                          y=portugal$mpw, committees=1)

portugal_train <- portugal[portugal$Year < 2018, ]
portugal_test  <- portugal[portugal$Year >= 2018, ]

portugal_cubist_test <- cubist(x=as.matrix(portugal_train[, c('Population', 'material_footprint')]),
                               y=portugal_train$mpw, committees=1)

portugal_test_preds <- predict(portugal_cubist_test,
                               newdata=portugal_test[, c('Population', 'material_footprint')],
                               neighbors=3)
portugal_test_residuals <- portugal_test$mpw - portugal_test_preds
portugal_centred_residuals <- portugal_test_residuals - mean(portugal_test_residuals)

portugal_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  portugal_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Portugal'&population_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Portugal'&material_footprint_forecasts$sim_id==i,'value']
  )
  portugal_forecast <- predict(portugal_cubist, newdata=portugal_newdata, neighbors=3)
  portugal_boot_residual <- sample(portugal_centred_residuals, size=1)
  portugal_temp <- data.frame(year=2023:2030, sim_id=i, value=portugal_forecast + portugal_boot_residual)
  portugal_metallic_forecasts <- rbind(portugal_metallic_forecasts, portugal_temp)
}

portugal_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                  lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  portugal_median <- median(portugal_metallic_forecasts[portugal_metallic_forecasts$year==i,]$value)
  portugal_lower  <- quantile(portugal_metallic_forecasts[portugal_metallic_forecasts$year==i,]$value, probs=0.025)
  portugal_upper  <- quantile(portugal_metallic_forecasts[portugal_metallic_forecasts$year==i,]$value, probs=0.975)
  portugal_median_pis[nrow(portugal_median_pis)+1,] <- c(i, portugal_median, portugal_lower, portugal_upper)
}

plot(y=portugal$mpw, x=portugal$Year, type='b',
     xlab='Year', ylab='%', main='Portugal Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(portugal$mpw), max(portugal_median_pis$upper_pi)))
lines(y=portugal_median_pis$median_forecast, x=portugal_median_pis$year, type='b', col='blue')
polygon(c(portugal_median_pis$year, rev(portugal_median_pis$year)),
        c(portugal_median_pis$upper_pi, rev(portugal_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#spain----
spain_svm <- svm(mpw~Population+GDP+exports+energy_consumption, data=spain, kernel='radial',
                 cost=5, gamma=0.05, epsilon=0.001)

spain_train <- spain[spain$Year < 2018, ]
spain_test  <- spain[spain$Year >= 2018, ]

spain_svm_test <- svm(mpw~Population+GDP+exports+energy_consumption, data=spain_train, kernel='radial',
                      cost=5, gamma=0.05, epsilon=0.001)

spain_test_preds <- predict(spain_svm_test,
                            newdata=spain_test[, c('Population', 'GDP', 'exports', 'energy_consumption')])
spain_test_residuals <- spain_test$mpw - spain_test_preds
spain_centred_residuals <- spain_test_residuals - mean(spain_test_residuals)

spain_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  spain_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Spain'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Spain'&gdp_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Spain'&exports_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Spain'&energy_consumption_forecasts$sim_id==i,'value']
  )
  spain_forecast <- predict(spain_svm, newdata=spain_newdata)
  spain_boot_residual <- sample(spain_centred_residuals, size=1)
  spain_temp <- data.frame(year=2023:2030, sim_id=i, value=spain_forecast + spain_boot_residual)
  spain_metallic_forecasts <- rbind(spain_metallic_forecasts, spain_temp)
}

spain_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                               lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  spain_median <- median(spain_metallic_forecasts[spain_metallic_forecasts$year==i,]$value)
  spain_lower  <- quantile(spain_metallic_forecasts[spain_metallic_forecasts$year==i,]$value, probs=0.025)
  spain_upper  <- quantile(spain_metallic_forecasts[spain_metallic_forecasts$year==i,]$value, probs=0.975)
  spain_median_pis[nrow(spain_median_pis)+1,] <- c(i, spain_median, spain_lower, spain_upper)
}

plot(y=spain$mpw, x=spain$Year, type='b',
     xlab='Year', ylab='%', main='Spain Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(spain$mpw), max(spain_median_pis$upper_pi)))
lines(y=spain_median_pis$median_forecast, x=spain_median_pis$year, type='b', col='blue')
polygon(c(spain_median_pis$year, rev(spain_median_pis$year)),
        c(spain_median_pis$upper_pi, rev(spain_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#sweden----
set.seed(123)
sweden_svm <- svm(mpw~GDP+Population, data=sweden, kernel='radial',
                  cost=1.5, gamma=0.05, epsilon=0.091)

sweden_train <- sweden[sweden$Year < 2018, ]
sweden_test  <- sweden[sweden$Year >= 2018, ]

set.seed(123)
sweden_svm_test <- svm(mpw~GDP+Population, data=sweden_train, kernel='radial',
                       cost=1.5, gamma=0.05, epsilon=0.091)

sweden_test_preds <- predict(sweden_svm_test,
                             newdata=sweden_test[, c('GDP', 'Population')])
sweden_test_residuals <- sweden_test$mpw - sweden_test_preds
sweden_centred_residuals <- sweden_test_residuals - mean(sweden_test_residuals)

sweden_metallic_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  sweden_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Sweden'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Sweden'&gdp_forecasts$sim_id==i,'value']
  )
  sweden_forecast <- predict(sweden_svm, newdata=sweden_newdata)
  sweden_boot_residual <- sample(sweden_centred_residuals, size=1)
  sweden_temp <- data.frame(year=2023:2030, sim_id=i, value=sweden_forecast + sweden_boot_residual)
  sweden_metallic_forecasts <- rbind(sweden_metallic_forecasts, sweden_temp)
}

sweden_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  sweden_median <- median(sweden_metallic_forecasts[sweden_metallic_forecasts$year==i,]$value)
  sweden_lower  <- quantile(sweden_metallic_forecasts[sweden_metallic_forecasts$year==i,]$value, probs=0.025)
  sweden_upper  <- quantile(sweden_metallic_forecasts[sweden_metallic_forecasts$year==i,]$value, probs=0.975)
  sweden_median_pis[nrow(sweden_median_pis)+1,] <- c(i, sweden_median, sweden_lower, sweden_upper)
}

plot(y=sweden$mpw, x=sweden$Year, type='b',
     xlab='Year', ylab='%', main='Sweden Metallic Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=sweden_median_pis$median_forecast, x=sweden_median_pis$year, type='b', col='blue')
polygon(c(sweden_median_pis$year, rev(sweden_median_pis$year)),
        c(sweden_median_pis$upper_pi, rev(sweden_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#comparing 2023 forecasts to actual value----
#austria
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Austria
austria_median_pis[austria_median_pis$year=='2023',]$median_forecast
austria_median_pis[austria_median_pis$year=='2023',]$lower_pi
austria_median_pis[austria_median_pis$year=='2023',]$upper_pi

#belgium
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Belgium
belgium_median_pis[belgium_median_pis$year=='2023',]$median_forecast
belgium_median_pis[belgium_median_pis$year=='2023',]$lower_pi
belgium_median_pis[belgium_median_pis$year=='2023',]$upper_pi

#denmark
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Denmark
denmark_median_pis[denmark_median_pis$year=='2023',]$median_forecast
denmark_median_pis[denmark_median_pis$year=='2023',]$lower_pi
denmark_median_pis[denmark_median_pis$year=='2023',]$upper_pi

#finland
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Finland
finland_median_pis[finland_median_pis$year=='2023',]$median_forecast
finland_median_pis[finland_median_pis$year=='2023',]$lower_pi
finland_median_pis[finland_median_pis$year=='2023',]$upper_pi

#france
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$France
france_median_pis[france_median_pis$year=='2023',]$median_forecast
france_median_pis[france_median_pis$year=='2023',]$lower_pi
france_median_pis[france_median_pis$year=='2023',]$upper_pi

#germany
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Germany
germany_median_pis[germany_median_pis$year=='2023',]$median_forecast
germany_median_pis[germany_median_pis$year=='2023',]$lower_pi
germany_median_pis[germany_median_pis$year=='2023',]$upper_pi

#ireland
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Ireland
ireland_median_pis[ireland_median_pis$year=='2023',]$median_forecast
ireland_median_pis[ireland_median_pis$year=='2023',]$lower_pi
ireland_median_pis[ireland_median_pis$year=='2023',]$upper_pi

#italy
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Italy
italy_median_pis[italy_median_pis$year=='2023',]$median_forecast
italy_median_pis[italy_median_pis$year=='2023',]$lower_pi
italy_median_pis[italy_median_pis$year=='2023',]$upper_pi

#luxembourg
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Luxembourg
luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$median_forecast
luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$lower_pi
luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$upper_pi

#netherlands
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Netherlands
netherlands_median_pis[netherlands_median_pis$year=='2023',]$median_forecast
netherlands_median_pis[netherlands_median_pis$year=='2023',]$lower_pi
netherlands_median_pis[netherlands_median_pis$year=='2023',]$upper_pi

#portugal
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Portugal
portugal_median_pis[portugal_median_pis$year=='2023',]$median_forecast
portugal_median_pis[portugal_median_pis$year=='2023',]$lower_pi
portugal_median_pis[portugal_median_pis$year=='2023',]$upper_pi

#spain
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Spain
spain_median_pis[spain_median_pis$year=='2023',]$median_forecast
spain_median_pis[spain_median_pis$year=='2023',]$lower_pi
spain_median_pis[spain_median_pis$year=='2023',]$upper_pi

#sweden
metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Sweden
sweden_median_pis[sweden_median_pis$year=='2023',]$median_forecast
sweden_median_pis[sweden_median_pis$year=='2023',]$lower_pi
sweden_median_pis[sweden_median_pis$year=='2023',]$upper_pi


#plot----
pdf("metallic_forecasts.pdf", width=12, height=16)
par(mfrow=c(7, 2),
    mar=c(4, 5, 2, 1),
    cex.main=2,
    cex.lab=1.5,
    cex.axis=1.5)

plot(y=austria$mpw, x=austria$Year, type='b',
     xlab='Year', ylab='%', main='Austria Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(austria$mpw), max(austria_median_pis$upper_pi)))
lines(y=austria_median_pis$median_forecast, x=austria_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Austria, col='red', pch=17)
polygon(c(austria_median_pis$year, rev(austria_median_pis$year)),
        c(austria_median_pis$upper_pi, rev(austria_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=belgium$mpw, x=belgium$Year, type='b',
     xlab='Year', ylab='%', main='Belgium Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(belgium$mpw), max(belgium_median_pis$upper_pi)))
lines(y=belgium_median_pis$median_forecast, x=belgium_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Belgium, col='red', pch=17)
polygon(c(belgium_median_pis$year, rev(belgium_median_pis$year)),
        c(belgium_median_pis$upper_pi, rev(belgium_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=denmark$mpw, x=denmark$Year, type='b',
     xlab='Year', ylab='%', main='Denmark Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(denmark$mpw), max(denmark_median_pis$upper_pi)))
lines(y=denmark_median_pis$median_forecast, x=denmark_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Denmark, col='red', pch=17)
polygon(c(denmark_median_pis$year, rev(denmark_median_pis$year)),
        c(denmark_median_pis$upper_pi, rev(denmark_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=finland$mpw, x=finland$Year, type='b',
     xlab='Year', ylab='%', main='Finland Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(finland$mpw), max(finland_median_pis$upper_pi)))
lines(y=finland_median_pis$median_forecast, x=finland_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Finland, col='red', pch=17)
polygon(c(finland_median_pis$year, rev(finland_median_pis$year)),
        c(finland_median_pis$upper_pi, rev(finland_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=france$mpw, x=france$Year, type='b',
     xlab='Year', ylab='%', main='France Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(france$mpw), max(france_median_pis$upper_pi)))
lines(y=france_median_pis$median_forecast, x=france_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$France, col='red', pch=17)
polygon(c(france_median_pis$year, rev(france_median_pis$year)),
        c(france_median_pis$upper_pi, rev(france_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=germany$mpw, x=germany$Year, type='b',
     xlab='Year', ylab='%', main='Germany Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(germany_median_pis$lower_pi), max(germany$mpw)))
lines(y=germany_median_pis$median_forecast, x=germany_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Germany, col='red', pch=17)
polygon(c(germany_median_pis$year, rev(germany_median_pis$year)),
        c(germany_median_pis$upper_pi, rev(germany_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=ireland$mpw, x=ireland$Year, type='b',
     xlab='Year', ylab='%', main='Ireland Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(ireland$mpw), max(ireland$mpw)))
lines(y=ireland_median_pis$median_forecast, x=ireland_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Ireland, col='red', pch=17)
polygon(c(ireland_median_pis$year, rev(ireland_median_pis$year)),
        c(ireland_median_pis$upper_pi, rev(ireland_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=italy$mpw, x=italy$Year, type='b',
     xlab='Year', ylab='%', main='Italy Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(italy$mpw), max(italy_median_pis$upper_pi)))
lines(y=italy_median_pis$median_forecast, x=italy_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Italy, col='red', pch=17)
polygon(c(italy_median_pis$year, rev(italy_median_pis$year)),
        c(italy_median_pis$upper_pi, rev(italy_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=luxembourg$mpw, x=luxembourg$Year, type='b',
     xlab='Year', ylab='%', main='Luxembourg Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(luxembourg$mpw), max(luxembourg_median_pis$upper_pi)))
lines(y=luxembourg_median_pis$median_forecast, x=luxembourg_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Luxembourg, col='red', pch=17)
polygon(c(luxembourg_median_pis$year, rev(luxembourg_median_pis$year)),
        c(luxembourg_median_pis$upper_pi, rev(luxembourg_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=netherlands$mpw, x=netherlands$Year, type='b',
     xlab='Year', ylab='%', main='Netherlands Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(netherlands$mpw), max(e=netherlands$mpw)))
lines(y=netherlands_median_pis$median_forecast, x=netherlands_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Netherlands, col='red', pch=17)
polygon(c(netherlands_median_pis$year, rev(netherlands_median_pis$year)),
        c(netherlands_median_pis$upper_pi, rev(netherlands_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=portugal$mpw, x=portugal$Year, type='b',
     xlab='Year', ylab='%', main='Portugal Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(portugal$mpw), max(portugal_median_pis$upper_pi)))
lines(y=portugal_median_pis$median_forecast, x=portugal_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Portugal, col='red', pch=17)
polygon(c(portugal_median_pis$year, rev(portugal_median_pis$year)),
        c(portugal_median_pis$upper_pi, rev(portugal_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=spain$mpw, x=spain$Year, type='b',
     xlab='Year', ylab='%', main='Spain Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(spain$mpw), max(spain_median_pis$upper_pi)))
lines(y=spain_median_pis$median_forecast, x=spain_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Spain, col='red', pch=17)
polygon(c(spain_median_pis$year, rev(spain_median_pis$year)),
        c(spain_median_pis$upper_pi, rev(spain_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=sweden$mpw, x=sweden$Year, type='b',
     xlab='Year', ylab='%', main='Sweden Metallic Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(sweden$mpw), max(metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Sweden)))
lines(y=sweden_median_pis$median_forecast, x=sweden_median_pis$year, type='b', col='blue')
points(x=2023, y=metallic_packaging_waste[metallic_packaging_waste$TIME=='2023',]$Sweden, col='red', pch=17)
polygon(c(sweden_median_pis$year, rev(sweden_median_pis$year)),
        c(sweden_median_pis$upper_pi, rev(sweden_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot.new()
legend("center", legend=c("Observed", "Median Forecast", "95% PI", "2023 Observed"),
       col=c("black", "blue", rgb(0, 0, 1, 0.2), "red"), lty=c(1, 1, NA, NA),
       pch=c(1, 1, 15, 17), pt.cex=c(1, 1, 2, 1), cex=2, bty="n")

dev.off()

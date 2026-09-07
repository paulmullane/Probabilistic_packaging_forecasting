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
glass_packaging_waste <- read_excel("Desktop/Packaging waste forecasting/Packaging Data/glass_packaging_waste.xlsx")
glass_packaging_waste$TIME <- as.double(glass_packaging_waste$TIME)

#training data
countries <- c("Austria", "Belgium", "Denmark", "Finland", "France", "Germany",
               "Ireland", "Italy", "Luxembourg", "Netherlands", "Portugal", "Spain", "Sweden")
vars <- c("Year", "Population", "GDP", "material_footprint", "energy_consumption", "co2", "exports")

modelling_data <- lapply(setNames(countries, tolower(countries)), function(ctry) {
  df <- read_excel(paste0("Desktop/Packaging waste forecasting/Country data/", ctry, ".xlsx")) |>
    dplyr::select(all_of(vars)) |>
    filter(Year >= 1997, Year <= 2022)
  df$gpw <- glass_packaging_waste[[ctry]][glass_packaging_waste$TIME <= 2022]
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

#austria----
set.seed(123)
austria_bayes_ridge <- bayesglm(gpw~exports, data=austria, family=gaussian)

# refit on pre-2018 data to compute genuine test set residuals
austria_train <- austria[austria$Year < 2018, ]
austria_test  <- austria[austria$Year >= 2018, ]

austria_bayes_ridge_test <- bayesglm(gpw~exports, data=austria_train, family=gaussian)

austria_test_preds <- predict(austria_bayes_ridge_test,
                              newdata=austria_test[, 'exports', drop=FALSE])
austria_test_residuals <- austria_test$gpw - austria_test_preds
austria_centred_residuals <- austria_test_residuals - mean(austria_test_residuals)

# blank dataframe to store the results
austria_glass_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                      sim_id=rep(1:1000, each=8), value=NA)

set.seed(123)
for(i in 1:1000){
  austria_newdata <- data.frame(exports=exports_forecasts[exports_forecasts$country=='Austria'&exports_forecasts$sim_id==i,'value'])
  austria_forecast <- predict(austria_bayes_ridge, newdata=austria_newdata)
  austria_boot_residual <- sample(austria_centred_residuals, size=1)
  austria_glass_forecasts$value[austria_glass_forecasts$sim_id==i] <- austria_forecast + austria_boot_residual
}

austria_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  austria_median <- median(austria_glass_forecasts[austria_glass_forecasts$year==i,]$value)
  austria_lower  <- quantile(austria_glass_forecasts[austria_glass_forecasts$year==i,]$value, probs=0.025)
  austria_upper  <- quantile(austria_glass_forecasts[austria_glass_forecasts$year==i,]$value, probs=0.975)
  austria_median_pis[nrow(austria_median_pis)+1,] <- c(i, austria_median, austria_lower, austria_upper)
}

plot(y=austria$gpw, x=austria$Year, type='b',
     xlab='Year', ylab='%', main='Austria Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=austria_median_pis$median_forecast, x=austria_median_pis$year, type='b', col='blue')
polygon(c(austria_median_pis$year, rev(austria_median_pis$year)),
        c(austria_median_pis$upper_pi, rev(austria_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#belgium----
set.seed(123)
belgium_svm <- svm(gpw~GDP, data=belgium, kernel='radial',
                   cost=5, gamma=0.5, epsilon=0.001)

belgium_train <- belgium[belgium$Year < 2018, ]
belgium_test  <- belgium[belgium$Year >= 2018, ]

set.seed(123)
belgium_svm_test <- svm(gpw~GDP, data=belgium_train, kernel='radial',
                        cost=5, gamma=0.5, epsilon=0.001)

belgium_test_preds <- predict(belgium_svm_test,
                              newdata=belgium_test[, 'GDP', drop=FALSE])
belgium_test_residuals <- belgium_test$gpw - belgium_test_preds
belgium_centred_residuals <- belgium_test_residuals - mean(belgium_test_residuals)

belgium_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  belgium_newdata <- data.frame(GDP=gdp_forecasts[gdp_forecasts$country=='Belgium'&gdp_forecasts$sim_id==i,'value'])
  belgium_forecast <- predict(belgium_svm, newdata=belgium_newdata)
  belgium_boot_residual <- sample(belgium_centred_residuals, size=1)
  belgium_temp <- data.frame(year=2023:2030, sim_id=i, value=belgium_forecast + belgium_boot_residual)
  belgium_glass_forecasts <- rbind(belgium_glass_forecasts, belgium_temp)
}

belgium_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  belgium_median <- median(belgium_glass_forecasts[belgium_glass_forecasts$year==i,]$value)
  belgium_lower  <- quantile(belgium_glass_forecasts[belgium_glass_forecasts$year==i,]$value, probs=0.025)
  belgium_upper  <- quantile(belgium_glass_forecasts[belgium_glass_forecasts$year==i,]$value, probs=0.975)
  belgium_median_pis[nrow(belgium_median_pis)+1,] <- c(i, belgium_median, belgium_lower, belgium_upper)
}

plot(y=belgium$gpw, x=belgium$Year, type='b',
     xlab='Year', ylab='%', main='Belgium Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=belgium_median_pis$median_forecast, x=belgium_median_pis$year, type='b', col='blue')
polygon(c(belgium_median_pis$year, rev(belgium_median_pis$year)),
        c(belgium_median_pis$upper_pi, rev(belgium_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#denmark----
set.seed(123)
denmark_svm <- svm(gpw~Population, data=denmark, kernel='radial',
                   cost=0.1, gamma=0.5, epsilon=0.041)

denmark_train <- denmark[denmark$Year < 2018, ]
denmark_test  <- denmark[denmark$Year >= 2018, ]

set.seed(123)
denmark_svm_test <- svm(gpw~Population, data=denmark_train, kernel='radial',
                        cost=0.1, gamma=0.5, epsilon=0.041)

denmark_test_preds <- predict(denmark_svm_test,
                              newdata=denmark_test[, 'Population', drop=FALSE])
denmark_test_residuals <- denmark_test$gpw - denmark_test_preds
denmark_centred_residuals <- denmark_test_residuals - mean(denmark_test_residuals)

denmark_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  denmark_newdata <- data.frame(Population=population_forecasts[population_forecasts$country=='Denmark'&population_forecasts$sim_id==i,'value'])
  denmark_forecast <- predict(denmark_svm, newdata=denmark_newdata)
  denmark_boot_residual <- sample(denmark_centred_residuals, size=1)
  denmark_temp <- data.frame(year=2023:2030, sim_id=i, value=denmark_forecast + denmark_boot_residual)
  denmark_glass_forecasts <- rbind(denmark_glass_forecasts, denmark_temp)
}

denmark_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  denmark_median <- median(denmark_glass_forecasts[denmark_glass_forecasts$year==i,]$value)
  denmark_lower  <- quantile(denmark_glass_forecasts[denmark_glass_forecasts$year==i,]$value, probs=0.025)
  denmark_upper  <- quantile(denmark_glass_forecasts[denmark_glass_forecasts$year==i,]$value, probs=0.975)
  denmark_median_pis[nrow(denmark_median_pis)+1,] <- c(i, denmark_median, denmark_lower, denmark_upper)
}

plot(y=denmark$gpw, x=denmark$Year, type='b',
     xlab='Year', ylab='%', main='Denmark Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=denmark_median_pis$median_forecast, x=denmark_median_pis$year, type='b', col='blue')
polygon(c(denmark_median_pis$year, rev(denmark_median_pis$year)),
        c(denmark_median_pis$upper_pi, rev(denmark_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#finland----
set.seed(123)
finland_svm <- svm(gpw~Population, data=finland, kernel='radial',
                   cost=4.6, gamma=0.1, epsilon=0.081)

finland_train <- finland[finland$Year < 2018, ]
finland_test  <- finland[finland$Year >= 2018, ]

set.seed(123)
finland_svm_test <- svm(gpw~Population, data=finland_train, kernel='radial',
                        cost=4.6, gamma=0.1, epsilon=0.081)

finland_test_preds <- predict(finland_svm_test,
                              newdata=finland_test[, 'Population', drop=FALSE])
finland_test_residuals <- finland_test$gpw - finland_test_preds
finland_centred_residuals <- finland_test_residuals - mean(finland_test_residuals)

finland_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  finland_newdata <- data.frame(Population=population_forecasts[population_forecasts$country=='Finland'&population_forecasts$sim_id==i,'value'])
  finland_forecast <- predict(finland_svm, newdata=finland_newdata)
  finland_boot_residual <- sample(finland_centred_residuals, size=1)
  finland_temp <- data.frame(year=2023:2030, sim_id=i, value=finland_forecast + finland_boot_residual)
  finland_glass_forecasts <- rbind(finland_glass_forecasts, finland_temp)
}

finland_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  finland_median <- median(finland_glass_forecasts[finland_glass_forecasts$year==i,]$value)
  finland_lower  <- quantile(finland_glass_forecasts[finland_glass_forecasts$year==i,]$value, probs=0.025)
  finland_upper  <- quantile(finland_glass_forecasts[finland_glass_forecasts$year==i,]$value, probs=0.975)
  finland_median_pis[nrow(finland_median_pis)+1,] <- c(i, finland_median, finland_lower, finland_upper)
}

plot(y=finland$gpw, x=finland$Year, type='b',
     xlab='Year', ylab='%', main='Finland Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(finland$gpw), max(finland_median_pis$upper_pi)))
lines(y=finland_median_pis$median_forecast, x=finland_median_pis$year, type='b', col='blue')
polygon(c(finland_median_pis$year, rev(finland_median_pis$year)),
        c(finland_median_pis$upper_pi, rev(finland_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#france----
set.seed(123)
france_bayes_ridge <- bayesglm(gpw~Population, data=france, family=gaussian)

france_train <- france[france$Year < 2018, ]
france_test  <- france[france$Year >= 2018, ]

france_bayes_ridge_test <- bayesglm(gpw~Population, data=france_train, family=gaussian)

france_test_preds <- predict(france_bayes_ridge_test,
                             newdata=france_test[, 'Population', drop=FALSE])
france_test_residuals <- france_test$gpw - france_test_preds
france_centred_residuals <- france_test_residuals - mean(france_test_residuals)

france_glass_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                     sim_id=rep(1:1000, each=8), value=NA)

set.seed(123)
for(i in 1:1000){
  france_newdata <- data.frame(Population=population_forecasts[population_forecasts$country=='France'&population_forecasts$sim_id==i,'value'])
  france_forecast <- predict(france_bayes_ridge, newdata=france_newdata)
  france_boot_residual <- sample(france_centred_residuals, size=1)
  france_glass_forecasts$value[france_glass_forecasts$sim_id==i] <- france_forecast + france_boot_residual
}

france_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  france_median <- median(france_glass_forecasts[france_glass_forecasts$year==i,]$value)
  france_lower  <- quantile(france_glass_forecasts[france_glass_forecasts$year==i,]$value, probs=0.025)
  france_upper  <- quantile(france_glass_forecasts[france_glass_forecasts$year==i,]$value, probs=0.975)
  france_median_pis[nrow(france_median_pis)+1,] <- c(i, france_median, france_lower, france_upper)
}

plot(y=france$gpw, x=france$Year, type='b',
     xlab='Year', ylab='%', main='France Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(france$gpw), max(france_median_pis$upper_pi)))
lines(y=france_median_pis$median_forecast, x=france_median_pis$year, type='b', col='blue')
polygon(c(france_median_pis$year, rev(france_median_pis$year)),
        c(france_median_pis$upper_pi, rev(france_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#germany----
set.seed(123)
germany_bayes_ridge <- bayesglm(gpw~Population, data=germany, family=gaussian)

germany_train <- germany[germany$Year < 2018, ]
germany_test  <- germany[germany$Year >= 2018, ]

germany_bayes_ridge_test <- bayesglm(gpw~Population, data=germany_train, family=gaussian)

germany_test_preds <- predict(germany_bayes_ridge_test,
                              newdata=germany_test[, 'Population', drop=FALSE])
germany_test_residuals <- germany_test$gpw - germany_test_preds
germany_centred_residuals <- germany_test_residuals - mean(germany_test_residuals)

germany_glass_forecasts <- data.frame(year=rep(2023:2030, times=1000),
                                      sim_id=rep(1:1000, each=8), value=NA)

set.seed(123)
for(i in 1:1000){
  germany_newdata <- data.frame(Population=population_forecasts[population_forecasts$country=='Germany'&population_forecasts$sim_id==i,'value'])
  germany_forecast <- predict(germany_bayes_ridge, newdata=germany_newdata)
  germany_boot_residual <- sample(germany_centred_residuals, size=1)
  germany_glass_forecasts$value[germany_glass_forecasts$sim_id==i] <- germany_forecast + germany_boot_residual
}

germany_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  germany_median <- median(germany_glass_forecasts[germany_glass_forecasts$year==i,]$value)
  germany_lower  <- quantile(germany_glass_forecasts[germany_glass_forecasts$year==i,]$value, probs=0.025)
  germany_upper  <- quantile(germany_glass_forecasts[germany_glass_forecasts$year==i,]$value, probs=0.975)
  germany_median_pis[nrow(germany_median_pis)+1,] <- c(i, germany_median, germany_lower, germany_upper)
}

plot(y=germany$gpw, x=germany$Year, type='b',
     xlab='Year', ylab='%', main='Germany Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(germany_median_pis$lower_pi), max(germany$gpw)))
lines(y=germany_median_pis$median_forecast, x=germany_median_pis$year, type='b', col='blue')
polygon(c(germany_median_pis$year, rev(germany_median_pis$year)),
        c(germany_median_pis$upper_pi, rev(germany_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#ireland----
set.seed(123)
ireland_svm <- svm(gpw~Population, data=ireland, kernel='radial',
                   cost=3.9, gamma=0.05, epsilon=0.081)

ireland_train <- ireland[ireland$Year < 2018, ]
ireland_test  <- ireland[ireland$Year >= 2018, ]

set.seed(123)
ireland_svm_test <- svm(gpw~Population, data=ireland_train, kernel='radial',
                        cost=3.9, gamma=0.05, epsilon=0.081)

ireland_test_preds <- predict(ireland_svm_test,
                              newdata=ireland_test[, 'Population', drop=FALSE])
ireland_test_residuals <- ireland_test$gpw - ireland_test_preds
ireland_centred_residuals <- ireland_test_residuals - mean(ireland_test_residuals)

ireland_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  ireland_newdata <- data.frame(Population=population_forecasts[population_forecasts$country=='Ireland'&population_forecasts$sim_id==i,'value'])
  ireland_forecast <- predict(ireland_svm, newdata=ireland_newdata)
  ireland_boot_residual <- sample(ireland_centred_residuals, size=1)
  ireland_temp <- data.frame(year=2023:2030, sim_id=i, value=ireland_forecast + ireland_boot_residual)
  ireland_glass_forecasts <- rbind(ireland_glass_forecasts, ireland_temp)
}

ireland_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                 lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  ireland_median <- median(ireland_glass_forecasts[ireland_glass_forecasts$year==i,]$value)
  ireland_lower  <- quantile(ireland_glass_forecasts[ireland_glass_forecasts$year==i,]$value, probs=0.025)
  ireland_upper  <- quantile(ireland_glass_forecasts[ireland_glass_forecasts$year==i,]$value, probs=0.975)
  ireland_median_pis[nrow(ireland_median_pis)+1,] <- c(i, ireland_median, ireland_lower, ireland_upper)
}

plot(y=ireland$gpw, x=ireland$Year, type='b',
     xlab='Year', ylab='%', main='Ireland Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(ireland$gpw), max(ireland$gpw)))
lines(y=ireland_median_pis$median_forecast, x=ireland_median_pis$year, type='b', col='blue')
polygon(c(ireland_median_pis$year, rev(ireland_median_pis$year)),
        c(ireland_median_pis$upper_pi, rev(ireland_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#italy----
italy_bayes_ridge <- bayesglm(gpw~Population+GDP+co2+energy_consumption, data=italy,
                              family=gaussian)

italy_train <- italy[italy$Year < 2018, ]
italy_test  <- italy[italy$Year >= 2018, ]

italy_bayes_ridge_test <- bayesglm(gpw~Population+GDP+co2+energy_consumption, data=italy_train,
                                   family=gaussian)

italy_test_preds <- predict(italy_bayes_ridge_test,
                            newdata=italy_test[, c('Population', 'GDP', 'co2', 'energy_consumption')])
italy_test_residuals <- italy_test$gpw - italy_test_preds
italy_centred_residuals <- italy_test_residuals - mean(italy_test_residuals)

italy_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  italy_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Italy'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Italy'&gdp_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Italy'&co2_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Italy'&energy_consumption_forecasts$sim_id==i,'value']
  )
  italy_forecast <- predict(italy_bayes_ridge, newdata=italy_newdata)
  italy_boot_residual <- sample(italy_centred_residuals, size=1)
  italy_temp <- data.frame(year=2023:2030, sim_id=i, value=italy_forecast + italy_boot_residual)
  italy_glass_forecasts <- rbind(italy_glass_forecasts, italy_temp)
}

italy_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                               lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  italy_median <- median(italy_glass_forecasts[italy_glass_forecasts$year==i,]$value)
  italy_lower  <- quantile(italy_glass_forecasts[italy_glass_forecasts$year==i,]$value, probs=0.025)
  italy_upper  <- quantile(italy_glass_forecasts[italy_glass_forecasts$year==i,]$value, probs=0.975)
  italy_median_pis[nrow(italy_median_pis)+1,] <- c(i, italy_median, italy_lower, italy_upper)
}

plot(y=italy$gpw, x=italy$Year, type='b',
     xlab='Year', ylab='%', main='Italy Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(italy$gpw), max(italy_median_pis$upper_pi)))
lines(y=italy_median_pis$median_forecast, x=italy_median_pis$year, type='b', col='blue')
polygon(c(italy_median_pis$year, rev(italy_median_pis$year)),
        c(italy_median_pis$upper_pi, rev(italy_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#luxembourg----
set.seed(123)
luxembourg_svm <- svm(gpw~Population+GDP+exports+energy_consumption, data=luxembourg, kernel='radial',
                      cost=3.5, gamma=0.05, epsilon=0.001)

luxembourg_train <- luxembourg[luxembourg$Year < 2018, ]
luxembourg_test  <- luxembourg[luxembourg$Year >= 2018, ]

set.seed(123)
luxembourg_svm_test <- svm(gpw~Population+GDP+exports+energy_consumption, data=luxembourg_train, kernel='radial',
                           cost=3.5, gamma=0.05, epsilon=0.001)

luxembourg_test_preds <- predict(luxembourg_svm_test,
                                 newdata=luxembourg_test[, c('Population', 'GDP', 'exports', 'energy_consumption')])
luxembourg_test_residuals <- luxembourg_test$gpw - luxembourg_test_preds
luxembourg_centred_residuals <- luxembourg_test_residuals - mean(luxembourg_test_residuals)

luxembourg_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  luxembourg_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Luxembourg'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Luxembourg'&gdp_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Luxembourg'&exports_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Luxembourg'&energy_consumption_forecasts$sim_id==i,'value']
  )
  luxembourg_forecast <- predict(luxembourg_svm, newdata=luxembourg_newdata)
  luxembourg_boot_residual <- sample(luxembourg_centred_residuals, size=1)
  luxembourg_temp <- data.frame(year=2023:2030, sim_id=i, value=luxembourg_forecast + luxembourg_boot_residual)
  luxembourg_glass_forecasts <- rbind(luxembourg_glass_forecasts, luxembourg_temp)
}

luxembourg_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                    lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  luxembourg_median <- median(luxembourg_glass_forecasts[luxembourg_glass_forecasts$year==i,]$value)
  luxembourg_lower  <- quantile(luxembourg_glass_forecasts[luxembourg_glass_forecasts$year==i,]$value, probs=0.025)
  luxembourg_upper  <- quantile(luxembourg_glass_forecasts[luxembourg_glass_forecasts$year==i,]$value, probs=0.975)
  luxembourg_median_pis[nrow(luxembourg_median_pis)+1,] <- c(i, luxembourg_median, luxembourg_lower, luxembourg_upper)
}

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b',
     xlab='Year', ylab='%', main='Luxembourg Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=luxembourg_median_pis$median_forecast, x=luxembourg_median_pis$year, type='b', col='blue')
polygon(c(luxembourg_median_pis$year, rev(luxembourg_median_pis$year)),
        c(luxembourg_median_pis$upper_pi, rev(luxembourg_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#netherlands----
set.seed(123)
netherlands_svm <- svm(gpw~GDP, data=netherlands, kernel='radial',
                       cost=1, gamma=0.05, epsilon=0.041)

netherlands_train <- netherlands[netherlands$Year < 2018, ]
netherlands_test  <- netherlands[netherlands$Year >= 2018, ]

set.seed(123)
netherlands_svm_test <- svm(gpw~GDP, data=netherlands_train, kernel='radial',
                            cost=1, gamma=0.05, epsilon=0.041)

netherlands_test_preds <- predict(netherlands_svm_test,
                                  newdata=netherlands_test[, 'GDP', drop=FALSE])
netherlands_test_residuals <- netherlands_test$gpw - netherlands_test_preds
netherlands_centred_residuals <- netherlands_test_residuals - mean(netherlands_test_residuals)

netherlands_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  netherlands_newdata <- data.frame(GDP=gdp_forecasts[gdp_forecasts$country=='Netherlands'&gdp_forecasts$sim_id==i,'value'])
  netherlands_forecast <- predict(netherlands_svm, newdata=netherlands_newdata)
  netherlands_boot_residual <- sample(netherlands_centred_residuals, size=1)
  netherlands_temp <- data.frame(year=2023:2030, sim_id=i, value=netherlands_forecast + netherlands_boot_residual)
  netherlands_glass_forecasts <- rbind(netherlands_glass_forecasts, netherlands_temp)
}

netherlands_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                     lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  netherlands_median <- median(netherlands_glass_forecasts[netherlands_glass_forecasts$year==i,]$value)
  netherlands_lower  <- quantile(netherlands_glass_forecasts[netherlands_glass_forecasts$year==i,]$value, probs=0.025)
  netherlands_upper  <- quantile(netherlands_glass_forecasts[netherlands_glass_forecasts$year==i,]$value, probs=0.975)
  netherlands_median_pis[nrow(netherlands_median_pis)+1,] <- c(i, netherlands_median, netherlands_lower, netherlands_upper)
}

plot(y=netherlands$gpw, x=netherlands$Year, type='b',
     xlab='Year', ylab='%', main='Netherlands Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=netherlands_median_pis$median_forecast, x=netherlands_median_pis$year, type='b', col='blue')
polygon(c(netherlands_median_pis$year, rev(netherlands_median_pis$year)),
        c(netherlands_median_pis$upper_pi, rev(netherlands_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#portugal----
set.seed(20229798)
portugal_svm <- svm(gpw~co2, data=portugal, kernel='radial',
                    cost=0.3, gamma=0.3, epsilon=0.071)

portugal_train <- portugal[portugal$Year < 2018, ]
portugal_test  <- portugal[portugal$Year >= 2018, ]

set.seed(20229798)
portugal_svm_test <- svm(gpw~co2, data=portugal_train, kernel='radial',
                         cost=0.3, gamma=0.3, epsilon=0.071)

portugal_test_preds <- predict(portugal_svm_test,
                               newdata=portugal_test[, 'co2', drop=FALSE])
portugal_test_residuals <- portugal_test$gpw - portugal_test_preds
portugal_centred_residuals <- portugal_test_residuals - mean(portugal_test_residuals)

portugal_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(20229798)
for(i in 1:1000){
  portugal_newdata <- data.frame(co2=co2_forecasts[co2_forecasts$country=='Portugal'&co2_forecasts$sim_id==i,'value'])
  portugal_forecast <- predict(portugal_svm, newdata=portugal_newdata)
  portugal_boot_residual <- sample(portugal_centred_residuals, size=1)
  portugal_temp <- data.frame(year=2023:2030, sim_id=i, value=portugal_forecast + portugal_boot_residual)
  portugal_glass_forecasts <- rbind(portugal_glass_forecasts, portugal_temp)
}

portugal_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                  lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  portugal_median <- median(portugal_glass_forecasts[portugal_glass_forecasts$year==i,]$value)
  portugal_lower  <- quantile(portugal_glass_forecasts[portugal_glass_forecasts$year==i,]$value, probs=0.025)
  portugal_upper  <- quantile(portugal_glass_forecasts[portugal_glass_forecasts$year==i,]$value, probs=0.975)
  portugal_median_pis[nrow(portugal_median_pis)+1,] <- c(i, portugal_median, portugal_lower, portugal_upper)
}

plot(y=portugal$gpw, x=portugal$Year, type='b',
     xlab='Year', ylab='%', main='Portugal Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=portugal_median_pis$median_forecast, x=portugal_median_pis$year, type='b', col='blue')
polygon(c(portugal_median_pis$year, rev(portugal_median_pis$year)),
        c(portugal_median_pis$upper_pi, rev(portugal_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#spain----
set.seed(123)
spain_svm <- svm(gpw~Population+GDP+co2+material_footprint+exports+energy_consumption, data=spain, kernel='radial',
                 cost=3.7, gamma=0.05, epsilon=0.001)

spain_train <- spain[spain$Year < 2018, ]
spain_test  <- spain[spain$Year >= 2018, ]

set.seed(123)
spain_svm_test <- svm(gpw~Population+GDP+co2+material_footprint+exports+energy_consumption, data=spain_train, kernel='radial',
                      cost=3.7, gamma=0.05, epsilon=0.001)

spain_test_preds <- predict(spain_svm_test,
                            newdata=spain_test[, c('Population', 'GDP', 'co2', 'material_footprint', 'exports', 'energy_consumption')])
spain_test_residuals <- spain_test$gpw - spain_test_preds
spain_centred_residuals <- spain_test_residuals - mean(spain_test_residuals)

spain_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  spain_newdata <- data.frame(
    Population=population_forecasts[population_forecasts$country=='Spain'&population_forecasts$sim_id==i,'value'],
    GDP=gdp_forecasts[gdp_forecasts$country=='Spain'&gdp_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Spain'&co2_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Spain'&material_footprint_forecasts$sim_id==i,'value'],
    exports=exports_forecasts[exports_forecasts$country=='Spain'&exports_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Spain'&energy_consumption_forecasts$sim_id==i,'value']
  )
  spain_forecast <- predict(spain_svm, newdata=spain_newdata)
  spain_boot_residual <- sample(spain_centred_residuals, size=1)
  spain_temp <- data.frame(year=2023:2030, sim_id=i, value=spain_forecast + spain_boot_residual)
  spain_glass_forecasts <- rbind(spain_glass_forecasts, spain_temp)
}

spain_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                               lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  spain_median <- median(spain_glass_forecasts[spain_glass_forecasts$year==i,]$value)
  spain_lower  <- quantile(spain_glass_forecasts[spain_glass_forecasts$year==i,]$value, probs=0.025)
  spain_upper  <- quantile(spain_glass_forecasts[spain_glass_forecasts$year==i,]$value, probs=0.975)
  spain_median_pis[nrow(spain_median_pis)+1,] <- c(i, spain_median, spain_lower, spain_upper)
}

plot(y=spain$gpw, x=spain$Year, type='b',
     xlab='Year', ylab='%', main='Spain Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=spain_median_pis$median_forecast, x=spain_median_pis$year, type='b', col='blue')
polygon(c(spain_median_pis$year, rev(spain_median_pis$year)),
        c(spain_median_pis$upper_pi, rev(spain_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


#sweden----
set.seed(123)
sweden_svm <- svm(gpw~GDP+co2+material_footprint+energy_consumption, data=sweden, kernel='radial',
                  cost=5, gamma=0.05, epsilon=0.001)

sweden_train <- sweden[sweden$Year < 2018, ]
sweden_test  <- sweden[sweden$Year >= 2018, ]

set.seed(123)
sweden_svm_test <- svm(gpw~GDP+co2+material_footprint+energy_consumption, data=sweden_train, kernel='radial',
                       cost=5, gamma=0.05, epsilon=0.001)

sweden_test_preds <- predict(sweden_svm_test,
                             newdata=sweden_test[, c('GDP', 'co2', 'material_footprint', 'energy_consumption')])
sweden_test_residuals <- sweden_test$gpw - sweden_test_preds
sweden_centred_residuals <- sweden_test_residuals - mean(sweden_test_residuals)

sweden_glass_forecasts <- data.frame(year=numeric(), sim_id=numeric(), value=numeric())

set.seed(123)
for(i in 1:1000){
  sweden_newdata <- data.frame(
    GDP=gdp_forecasts[gdp_forecasts$country=='Sweden'&gdp_forecasts$sim_id==i,'value'],
    co2=co2_forecasts[co2_forecasts$country=='Sweden'&co2_forecasts$sim_id==i,'value'],
    energy_consumption=energy_consumption_forecasts[energy_consumption_forecasts$country=='Sweden'&energy_consumption_forecasts$sim_id==i,'value'],
    material_footprint=material_footprint_forecasts[material_footprint_forecasts$country=='Sweden'&material_footprint_forecasts$sim_id==i,'value']
  )
  sweden_forecast <- predict(sweden_svm, newdata=sweden_newdata)
  sweden_boot_residual <- sample(sweden_centred_residuals, size=1)
  sweden_temp <- data.frame(year=2023:2030, sim_id=i, value=sweden_forecast + sweden_boot_residual)
  sweden_glass_forecasts <- rbind(sweden_glass_forecasts, sweden_temp)
}

sweden_median_pis <- data.frame(year=numeric(), median_forecast=numeric(),
                                lower_pi=numeric(), upper_pi=numeric())
for(i in 2023:2030){
  sweden_median <- median(sweden_glass_forecasts[sweden_glass_forecasts$year==i,]$value)
  sweden_lower  <- quantile(sweden_glass_forecasts[sweden_glass_forecasts$year==i,]$value, probs=0.025)
  sweden_upper  <- quantile(sweden_glass_forecasts[sweden_glass_forecasts$year==i,]$value, probs=0.975)
  sweden_median_pis[nrow(sweden_median_pis)+1,] <- c(i, sweden_median, sweden_lower, sweden_upper)
}

plot(y=sweden$gpw, x=sweden$Year, type='b',
     xlab='Year', ylab='%', main='Sweden Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=sweden_median_pis$median_forecast, x=sweden_median_pis$year, type='b', col='blue')
polygon(c(sweden_median_pis$year, rev(sweden_median_pis$year)),
        c(sweden_median_pis$upper_pi, rev(sweden_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)



#comparing 2023 forecasts to actual value----
#austria
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Austria
austria_median_pis[austria_median_pis$year=='2023',]$median_forecast
austria_median_pis[austria_median_pis$year=='2023',]$lower_pi
austria_median_pis[austria_median_pis$year=='2023',]$upper_pi

#belgium
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Belgium
belgium_median_pis[belgium_median_pis$year=='2023',]$median_forecast
belgium_median_pis[belgium_median_pis$year=='2023',]$lower_pi
belgium_median_pis[belgium_median_pis$year=='2023',]$upper_pi

#denmark
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Denmark
denmark_median_pis[denmark_median_pis$year=='2023',]$median_forecast
denmark_median_pis[denmark_median_pis$year=='2023',]$lower_pi
denmark_median_pis[denmark_median_pis$year=='2023',]$upper_pi

#finland
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Finland
finland_median_pis[finland_median_pis$year=='2023',]$median_forecast
finland_median_pis[finland_median_pis$year=='2023',]$lower_pi
finland_median_pis[finland_median_pis$year=='2023',]$upper_pi

#france
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$France
france_median_pis[france_median_pis$year=='2023',]$median_forecast
france_median_pis[france_median_pis$year=='2023',]$lower_pi
france_median_pis[france_median_pis$year=='2023',]$upper_pi

#germany
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Germany
germany_median_pis[germany_median_pis$year=='2023',]$median_forecast
germany_median_pis[germany_median_pis$year=='2023',]$lower_pi
germany_median_pis[germany_median_pis$year=='2023',]$upper_pi

#ireland
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Ireland
ireland_median_pis[ireland_median_pis$year=='2023',]$median_forecast
ireland_median_pis[ireland_median_pis$year=='2023',]$lower_pi
ireland_median_pis[ireland_median_pis$year=='2023',]$upper_pi

#italy
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Italy
italy_median_pis[italy_median_pis$year=='2023',]$median_forecast
italy_median_pis[italy_median_pis$year=='2023',]$lower_pi
italy_median_pis[italy_median_pis$year=='2023',]$upper_pi

#luxembourg
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Luxembourg
luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$median_forecast
luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$lower_pi
luxembourg_median_pis[luxembourg_median_pis$year=='2023',]$upper_pi

#netherlands
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Netherlands
netherlands_median_pis[netherlands_median_pis$year=='2023',]$median_forecast
netherlands_median_pis[netherlands_median_pis$year=='2023',]$lower_pi
netherlands_median_pis[netherlands_median_pis$year=='2023',]$upper_pi

#portugal
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Portugal
portugal_median_pis[portugal_median_pis$year=='2023',]$median_forecast
portugal_median_pis[portugal_median_pis$year=='2023',]$lower_pi
portugal_median_pis[portugal_median_pis$year=='2023',]$upper_pi

#spain
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Spain
spain_median_pis[spain_median_pis$year=='2023',]$median_forecast
spain_median_pis[spain_median_pis$year=='2023',]$lower_pi
spain_median_pis[spain_median_pis$year=='2023',]$upper_pi

#sweden
glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Sweden
sweden_median_pis[sweden_median_pis$year=='2023',]$median_forecast
sweden_median_pis[sweden_median_pis$year=='2023',]$lower_pi
sweden_median_pis[sweden_median_pis$year=='2023',]$upper_pi

#plot----
pdf("glass_forecasts.pdf", width=12, height=16)
par(mfrow=c(7, 2),
    mar=c(4, 5, 2, 1),
    cex.main=2,
    cex.lab=1.5,
    cex.axis=1.5)

plot(y=austria$gpw, x=austria$Year, type='b',
     xlab='Year', ylab='%', main='Austria Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=austria_median_pis$median_forecast, x=austria_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Austria, col='red', pch=17)
polygon(c(austria_median_pis$year, rev(austria_median_pis$year)),
        c(austria_median_pis$upper_pi, rev(austria_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)


plot(y=belgium$gpw, x=belgium$Year, type='b',
     xlab='Year', ylab='%', main='Belgium Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=belgium_median_pis$median_forecast, x=belgium_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Belgium, col='red', pch=17)
polygon(c(belgium_median_pis$year, rev(belgium_median_pis$year)),
        c(belgium_median_pis$upper_pi, rev(belgium_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=denmark$gpw, x=denmark$Year, type='b',
     xlab='Year', ylab='%', main='Denmark Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=denmark_median_pis$median_forecast, x=denmark_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Denmark, col='red', pch=17)
polygon(c(denmark_median_pis$year, rev(denmark_median_pis$year)),
        c(denmark_median_pis$upper_pi, rev(denmark_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=finland$gpw, x=finland$Year, type='b',
     xlab='Year', ylab='%', main='Finland Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(finland$gpw), max(finland_median_pis$upper_pi)))
lines(y=finland_median_pis$median_forecast, x=finland_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Finland, col='red', pch=17)
polygon(c(finland_median_pis$year, rev(finland_median_pis$year)),
        c(finland_median_pis$upper_pi, rev(finland_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=france$gpw, x=france$Year, type='b',
     xlab='Year', ylab='%', main='France Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(france$gpw), max(france_median_pis$upper_pi)))
lines(y=france_median_pis$median_forecast, x=france_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$France, col='red', pch=17)
polygon(c(france_median_pis$year, rev(france_median_pis$year)),
        c(france_median_pis$upper_pi, rev(france_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=germany$gpw, x=germany$Year, type='b',
     xlab='Year', ylab='%', main='Germany Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(germany_median_pis$lower_pi), max(germany_median_pis$upper_pi)))
lines(y=germany_median_pis$median_forecast, x=germany_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Germany, col='red', pch=17)
polygon(c(germany_median_pis$year, rev(germany_median_pis$year)),
        c(germany_median_pis$upper_pi, rev(germany_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=ireland$gpw, x=ireland$Year, type='b',
     xlab='Year', ylab='%', main='Ireland Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(ireland$gpw), max(ireland$gpw)))
lines(y=ireland_median_pis$median_forecast, x=ireland_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Ireland, col='red', pch=17)
polygon(c(ireland_median_pis$year, rev(ireland_median_pis$year)),
        c(ireland_median_pis$upper_pi, rev(ireland_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=italy$gpw, x=italy$Year, type='b',
     xlab='Year', ylab='%', main='Italy Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(italy$gpw), max(italy_median_pis$upper_pi)))
lines(y=italy_median_pis$median_forecast, x=italy_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Italy, col='red', pch=17)
polygon(c(italy_median_pis$year, rev(italy_median_pis$year)),
        c(italy_median_pis$upper_pi, rev(italy_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=luxembourg$gpw, x=luxembourg$Year, type='b',
     xlab='Year', ylab='%', main='Luxembourg Glass Packaging Recycling',
     xlim=c(1997, 2030), ylim=c(min(luxembourg$gpw), max(luxembourg_median_pis$upper_pi)))
lines(y=luxembourg_median_pis$median_forecast, x=luxembourg_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Luxembourg, col='red', pch=17)
polygon(c(luxembourg_median_pis$year, rev(luxembourg_median_pis$year)),
        c(luxembourg_median_pis$upper_pi, rev(luxembourg_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=netherlands$gpw, x=netherlands$Year, type='b',
     xlab='Year', ylab='%', main='Netherlands Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=netherlands_median_pis$median_forecast, x=netherlands_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Netherlands, col='red', pch=17)
polygon(c(netherlands_median_pis$year, rev(netherlands_median_pis$year)),
        c(netherlands_median_pis$upper_pi, rev(netherlands_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=portugal$gpw, x=portugal$Year, type='b',
     xlab='Year', ylab='%', main='Portugal Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=portugal_median_pis$median_forecast, x=portugal_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Portugal, col='red', pch=17)
polygon(c(portugal_median_pis$year, rev(portugal_median_pis$year)),
        c(portugal_median_pis$upper_pi, rev(portugal_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=spain$gpw, x=spain$Year, type='b',
     xlab='Year', ylab='%', main='Spain Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=spain_median_pis$median_forecast, x=spain_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Spain, col='red', pch=17)
polygon(c(spain_median_pis$year, rev(spain_median_pis$year)),
        c(spain_median_pis$upper_pi, rev(spain_median_pis$lower_pi)),
        col=rgb(0, 0, 1, 0.2), border=NA)
#legend('topleft', legend=c('Observed', 'Median Forecast', '95% PI'),
#       col=c('black', 'blue', rgb(0,0,1,0.2)), lty=1)

plot(y=sweden$gpw, x=sweden$Year, type='b',
     xlab='Year', ylab='%', main='Sweden Glass Packaging Recycling',
     xlim=c(1997, 2030))
lines(y=sweden_median_pis$median_forecast, x=sweden_median_pis$year, type='b', col='blue')
points(x=2023, y=glass_packaging_waste[glass_packaging_waste$TIME=='2023',]$Sweden, col='red', pch=17)
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

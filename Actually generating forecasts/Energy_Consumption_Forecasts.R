#packages----
library(readxl)
library(dplyr)
library(Metrics)
library(forecast)
library(mgcv)

#read in the data----
Austria <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Austria.xlsx")[,c('Year', 'energy_consumption')])
Belgium <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Belgium.xlsx")[,c('Year', 'energy_consumption')])
Denmark<- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Denmark.xlsx")[,c('Year', 'energy_consumption')])
Finland <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Finland.xlsx")[,c('Year', 'energy_consumption')])
France <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/France.xlsx")[,c('Year', 'energy_consumption')])
Germany <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Germany.xlsx")[,c('Year', 'energy_consumption')])
Ireland <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Ireland.xlsx")[,c('Year', 'energy_consumption')])
Italy <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Italy.xlsx")[,c('Year', 'energy_consumption')])
Luxembourg <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Luxembourg.xlsx")[,c('Year', 'energy_consumption')])
Netherlands <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Netherlands.xlsx")[,c('Year', 'energy_consumption')])
Portugal <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Portugal.xlsx")[,c('Year', 'energy_consumption')])
Spain <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Spain.xlsx")[,c('Year', 'energy_consumption')])
Sweden <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Sweden.xlsx")[,c('Year', 'energy_consumption')])

#create empty dataframe----
energy_consumption_forecasts <- data.frame(country=character(), model=character(), 
                                           variable=character(), sim_id=numeric())

#Austria----
Austria_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Austria, method="REML")
Austria_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Austria_Xp <- predict(Austria_gam, newdata=Austria_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Austria_coefs <- rmvn(1000, coef(Austria_gam), vcov(Austria_gam))

#compute paths: each row is a draw, each column a year
Austria_paths <- t(Austria_Xp %*% t(Austria_coefs))

#add residual noise (sigma estimated from model)
Austria_sigma <- sqrt(Austria_gam$sig2)
Austria_paths <- Austria_paths+matrix(rnorm(8*1000, 0, Austria_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Austria", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Austria_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Austria_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Austria", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Austria_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Austria Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Austria_energy_consumption[Austria_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Belgium----
Belgium_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Belgium, method="REML")
Belgium_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Belgium_Xp <- predict(Belgium_gam, newdata=Belgium_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Belgium_coefs <- rmvn(1000, coef(Belgium_gam), vcov(Belgium_gam))

#compute paths: each row is a draw, each column a year
Belgium_paths <- t(Belgium_Xp %*% t(Belgium_coefs))

#add residual noise (sigma estimated from model)
Belgium_sigma <- sqrt(Belgium_gam$sig2)
Belgium_paths <- Belgium_paths+matrix(rnorm(8*1000, 0, Belgium_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Belgium", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Belgium_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Belgium_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Belgium", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Belgium_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Belgium Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Belgium_energy_consumption[Belgium_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Denmark----
Denmark_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Denmark, method="REML")
Denmark_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Denmark_Xp <- predict(Denmark_gam, newdata=Denmark_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Denmark_coefs <- rmvn(1000, coef(Denmark_gam), vcov(Denmark_gam))

#compute paths: each row is a draw, each column a year
Denmark_paths <- t(Denmark_Xp %*% t(Denmark_coefs))

#add residual noise (sigma estimated from model)
Denmark_sigma <- sqrt(Denmark_gam$sig2)
Denmark_paths <- Denmark_paths+matrix(rnorm(8*1000, 0, Denmark_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Denmark", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Denmark_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Denmark_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Denmark", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Denmark_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Denmark Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Denmark_energy_consumption[Denmark_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Finland----
Finland_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Finland, method="REML")
Finland_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Finland_Xp <- predict(Finland_gam, newdata=Finland_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Finland_coefs <- rmvn(1000, coef(Finland_gam), vcov(Finland_gam))

#compute paths: each row is a draw, each column a year
Finland_paths <- t(Finland_Xp %*% t(Finland_coefs))

#add residual noise (sigma estimated from model)
Finland_sigma <- sqrt(Finland_gam$sig2)
Finland_paths <- Finland_paths+matrix(rnorm(8*1000, 0, Finland_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Finland", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Finland_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Finland_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Finland", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Finland_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Finland Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Finland_energy_consumption[Finland_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#France----
France_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=France, method="REML")
France_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
France_Xp <- predict(France_gam, newdata=France_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
France_coefs <- rmvn(1000, coef(France_gam), vcov(France_gam))

#compute paths: each row is a draw, each column a year
France_paths <- t(France_Xp %*% t(France_coefs))

#add residual noise (sigma estimated from model)
France_sigma <- sqrt(France_gam$sig2)
France_paths <- France_paths+matrix(rnorm(8*1000, 0, France_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "France", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(France_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
France_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "France", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(France_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="France Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- France_energy_consumption[France_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Germany----
Germany_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Germany, method="REML")
Germany_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Germany_Xp <- predict(Germany_gam, newdata=Germany_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Germany_coefs <- rmvn(1000, coef(Germany_gam), vcov(Germany_gam))

#compute paths: each row is a draw, each column a year
Germany_paths <- t(Germany_Xp %*% t(Germany_coefs))

#add residual noise (sigma estimated from model)
Germany_sigma <- sqrt(Germany_gam$sig2)
Germany_paths <- Germany_paths+matrix(rnorm(8*1000, 0, Germany_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Germany", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Germany_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Germany_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Germany", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Germany_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Germany Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Germany_energy_consumption[Germany_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Ireland----
Ireland_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Ireland, method="REML")
Ireland_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Ireland_Xp <- predict(Ireland_gam, newdata=Ireland_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Ireland_coefs <- rmvn(1000, coef(Ireland_gam), vcov(Ireland_gam))

#compute paths: each row is a draw, each column a year
Ireland_paths <- t(Ireland_Xp %*% t(Ireland_coefs))

#add residual noise (sigma estimated from model)
Ireland_sigma <- sqrt(Ireland_gam$sig2)
Ireland_paths <- Ireland_paths+matrix(rnorm(8*1000, 0, Ireland_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Ireland", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Ireland_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Ireland_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Ireland", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Ireland_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Ireland Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Ireland_energy_consumption[Ireland_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Italy----
Italy_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Italy, method="REML")
Italy_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Italy_Xp <- predict(Italy_gam, newdata=Italy_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Italy_coefs <- rmvn(1000, coef(Italy_gam), vcov(Italy_gam))

#compute paths: each row is a draw, each column a year
Italy_paths <- t(Italy_Xp %*% t(Italy_coefs))

#add residual noise (sigma estimated from model)
Italy_sigma <- sqrt(Italy_gam$sig2)
Italy_paths <- Italy_paths+matrix(rnorm(8*1000, 0, Italy_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Italy", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Italy_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Italy_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Italy", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Italy_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Italy Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Italy_energy_consumption[Italy_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Luxembourg----
Luxembourg_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Luxembourg, method="REML")
Luxembourg_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Luxembourg_Xp <- predict(Luxembourg_gam, newdata=Luxembourg_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Luxembourg_coefs <- rmvn(1000, coef(Luxembourg_gam), vcov(Luxembourg_gam))

#compute paths: each row is a draw, each column a year
Luxembourg_paths <- t(Luxembourg_Xp %*% t(Luxembourg_coefs))

#add residual noise (sigma estimated from model)
Luxembourg_sigma <- sqrt(Luxembourg_gam$sig2)
Luxembourg_paths <- Luxembourg_paths+matrix(rnorm(8*1000, 0, Luxembourg_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Luxembourg", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Luxembourg_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Luxembourg_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Luxembourg", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Luxembourg_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Luxembourg Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Luxembourg_energy_consumption[Luxembourg_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Netherlands----
Netherlands_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Netherlands, method="REML")
Netherlands_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Netherlands_Xp <- predict(Netherlands_gam, newdata=Netherlands_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Netherlands_coefs <- rmvn(1000, coef(Netherlands_gam), vcov(Netherlands_gam))

#compute paths: each row is a draw, each column a year
Netherlands_paths <- t(Netherlands_Xp %*% t(Netherlands_coefs))

#add residual noise (sigma estimated from model)
Netherlands_sigma <- sqrt(Netherlands_gam$sig2)
Netherlands_paths <- Netherlands_paths+matrix(rnorm(8*1000, 0, Netherlands_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Netherlands", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Netherlands_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Netherlands_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Netherlands", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Netherlands_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Netherlands Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Netherlands_energy_consumption[Netherlands_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Portugal----
Portugal_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Portugal, method="REML")
Portugal_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Portugal_Xp <- predict(Portugal_gam, newdata=Portugal_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Portugal_coefs <- rmvn(1000, coef(Portugal_gam), vcov(Portugal_gam))

#compute paths: each row is a draw, each column a year
Portugal_paths <- t(Portugal_Xp %*% t(Portugal_coefs))

#add residual noise (sigma estimated from model)
Portugal_sigma <- sqrt(Portugal_gam$sig2)
Portugal_paths <- Portugal_paths+matrix(rnorm(8*1000, 0, Portugal_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Portugal", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Portugal_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Portugal_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Portugal", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Portugal_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Portugal Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Portugal_energy_consumption[Portugal_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Spain----
Spain_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Spain, method="REML")
Spain_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Spain_Xp <- predict(Spain_gam, newdata=Spain_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Spain_coefs <- rmvn(1000, coef(Spain_gam), vcov(Spain_gam))

#compute paths: each row is a draw, each column a year
Spain_paths <- t(Spain_Xp %*% t(Spain_coefs))

#add residual noise (sigma estimated from model)
Spain_sigma <- sqrt(Spain_gam$sig2)
Spain_paths <- Spain_paths+matrix(rnorm(8*1000, 0, Spain_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Spain", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Spain_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Spain_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Spain", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Spain_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Spain Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Spain_energy_consumption[Spain_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Sweden----
Sweden_gam <- mgcv::gam(energy_consumption~s(Year, k=8), data=Sweden, method="REML")
Sweden_new_data <- data.frame(Year = 2023:2030)

#draw 1000 samples from the posterior predictive distribution
set.seed(20229798)
Sweden_Xp <- predict(Sweden_gam, newdata=Sweden_new_data, type="lpmatrix") # covariance matrix of the linear predictor

#draw coefficients from posterior of the GAM
Sweden_coefs <- rmvn(1000, coef(Sweden_gam), vcov(Sweden_gam))

#compute paths: each row is a draw, each column a year
Sweden_paths <- t(Sweden_Xp %*% t(Sweden_coefs))

#add residual noise (sigma estimated from model)
Sweden_sigma <- sqrt(Sweden_gam$sig2)
Sweden_paths <- Sweden_paths+matrix(rnorm(8*1000, 0, Sweden_sigma), nrow=1000, ncol=8)

#change to long format and append to dataframe
energy_consumption_rows <- data.frame(country = "Sweden", model = "GAM", 
                                      variable = "energy_consumption", sim_id=rep(1:1000, each=8),
                                      year=rep(2023:2030, times=1000), value=as.vector(Sweden_paths))
energy_consumption_forecasts <- rbind(energy_consumption_forecasts, energy_consumption_rows)

#plot forecasted paths
Sweden_energy_consumption <- energy_consumption_forecasts[energy_consumption_forecasts$country == "Sweden", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Sweden_energy_consumption$value),
     xlab="Year", ylab="Energy Consumption", main="Sweden Energy Consumption - Simulated Paths")
for(i in 1:1000){
  sim_data <- Sweden_energy_consumption[Sweden_energy_consumption$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#save locally as a csv----
write.csv(energy_consumption_forecasts, file='energy_consumption_forecasts.csv')

#packages----
library(readxl)
library(dplyr)
library(Metrics)
library(forecast)

#read in the data----
Austria <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Austria.xlsx")[,c('Year', 'exports')])
Belgium <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Belgium.xlsx")[,c('Year', 'exports')])
Denmark<- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Denmark.xlsx")[,c('Year', 'exports')])
Finland <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Finland.xlsx")[,c('Year', 'exports')])
France <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/France.xlsx")[,c('Year', 'exports')])
Germany <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Germany.xlsx")[,c('Year', 'exports')])
Ireland <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Ireland.xlsx")[,c('Year', 'exports')])
Italy <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Italy.xlsx")[,c('Year', 'exports')])
Luxembourg <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Luxembourg.xlsx")[,c('Year', 'exports')])
Netherlands <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Netherlands.xlsx")[,c('Year', 'exports')])
Portugal <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Portugal.xlsx")[,c('Year', 'exports')])
Spain <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Spain.xlsx")[,c('Year', 'exports')])
Sweden <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Sweden.xlsx")[,c('Year', 'exports')])

#create empty dataframe----
exports_forecasts <- data.frame(country=character(), model=character(), 
                                variable=character(), sim_id=numeric(),
                                year=numeric(),value=numeric())

#Austria----
Austria_theta <- thetaf(Austria$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Austria_sigma <- sd(Austria_theta$residuals)
Austria_alpha <- Austria_theta$model$alpha
Austria_drift <- Austria_theta$model$drift

#simulate paths manually
Austria_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Austria$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Austria_sigma)
    path[t] <- ell+Austria_drift*t+eps
    ell     <- Austria_alpha*path[t]+(1-Austria_alpha)*ell
  }
  Austria_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Austria", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Austria_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Austria_exports <- exports_forecasts[exports_forecasts$Country == "Austria", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Austria_exports$value),xlab="Year", 
     ylab="exports", main="Austria exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Austria_exports[Austria_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Belgium----
Belgium_theta <- thetaf(Belgium$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Belgium_sigma <- sd(Belgium_theta$residuals)
Belgium_alpha <- Belgium_theta$model$alpha
Belgium_drift <- Belgium_theta$model$drift

#simulate paths manually
Belgium_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Belgium$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Belgium_sigma)
    path[t] <- ell+Belgium_drift*t+eps
    ell     <- Belgium_alpha*path[t]+(1-Belgium_alpha)*ell
  }
  Belgium_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Belgium", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Belgium_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Belgium_exports <- exports_forecasts[exports_forecasts$Country == "Belgium", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Belgium_exports$value),xlab="Year", 
     ylab="exports", main="Belgium exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Belgium_exports[Belgium_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Denmark----
Denmark_theta <- thetaf(Denmark$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Denmark_sigma <- sd(Denmark_theta$residuals)
Denmark_alpha <- Denmark_theta$model$alpha
Denmark_drift <- Denmark_theta$model$drift

#simulate paths manually
Denmark_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Denmark$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Denmark_sigma)
    path[t] <- ell+Denmark_drift*t+eps
    ell     <- Denmark_alpha*path[t]+(1-Denmark_alpha)*ell
  }
  Denmark_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Denmark", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Denmark_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Denmark_exports <- exports_forecasts[exports_forecasts$Country == "Denmark", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Denmark_exports$value),xlab="Year", 
     ylab="exports", main="Denmark exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Denmark_exports[Denmark_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Finland----
Finland_theta <- thetaf(Finland$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Finland_sigma <- sd(Finland_theta$residuals)
Finland_alpha <- Finland_theta$model$alpha
Finland_drift <- Finland_theta$model$drift

#simulate paths manually
Finland_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Finland$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Finland_sigma)
    path[t] <- ell+Finland_drift*t+eps
    ell     <- Finland_alpha*path[t]+(1-Finland_alpha)*ell
  }
  Finland_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Finland", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Finland_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Finland_exports <- exports_forecasts[exports_forecasts$Country == "Finland", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Finland_exports$value),xlab="Year", 
     ylab="exports", main="Finland exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Finland_exports[Finland_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#France----
France_theta <- thetaf(France$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
France_sigma <- sd(France_theta$residuals)
France_alpha <- France_theta$model$alpha
France_drift <- France_theta$model$drift

#simulate paths manually
France_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(France$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, France_sigma)
    path[t] <- ell+France_drift*t+eps
    ell     <- France_alpha*path[t]+(1-France_alpha)*ell
  }
  France_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "France", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(France_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
France_exports <- exports_forecasts[exports_forecasts$Country == "France", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(France_exports$value),xlab="Year", 
     ylab="exports", main="France exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- France_exports[France_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Germany----
Germany_theta <- thetaf(Germany$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Germany_sigma <- sd(Germany_theta$residuals)
Germany_alpha <- Germany_theta$model$alpha
Germany_drift <- Germany_theta$model$drift

#simulate paths manually
Germany_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Germany$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Germany_sigma)
    path[t] <- ell+Germany_drift*t+eps
    ell     <- Germany_alpha*path[t]+(1-Germany_alpha)*ell
  }
  Germany_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Germany", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Germany_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Germany_exports <- exports_forecasts[exports_forecasts$Country == "Germany", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Germany_exports$value),xlab="Year", 
     ylab="exports", main="Germany exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Germany_exports[Germany_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Ireland----
Ireland_theta <- thetaf(Ireland$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Ireland_sigma <- sd(Ireland_theta$residuals)
Ireland_alpha <- Ireland_theta$model$alpha
Ireland_drift <- Ireland_theta$model$drift

#simulate paths manually
Ireland_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Ireland$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Ireland_sigma)
    path[t] <- ell+Ireland_drift*t+eps
    ell     <- Ireland_alpha*path[t]+(1-Ireland_alpha)*ell
  }
  Ireland_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Ireland", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Ireland_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Ireland_exports <- exports_forecasts[exports_forecasts$Country == "Ireland", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Ireland_exports$value),xlab="Year", 
     ylab="exports", main="Ireland exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Ireland_exports[Ireland_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Italy----
Italy_theta <- thetaf(Italy$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Italy_sigma <- sd(Italy_theta$residuals)
Italy_alpha <- Italy_theta$model$alpha
Italy_drift <- Italy_theta$model$drift

#simulate paths manually
Italy_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Italy$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Italy_sigma)
    path[t] <- ell+Italy_drift*t+eps
    ell     <- Italy_alpha*path[t]+(1-Italy_alpha)*ell
  }
  Italy_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Italy", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Italy_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Italy_exports <- exports_forecasts[exports_forecasts$Country == "Italy", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Italy_exports$value),xlab="Year", 
     ylab="exports", main="Italy exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Italy_exports[Italy_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Luxembourg----
Luxembourg_theta <- thetaf(Luxembourg$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Luxembourg_sigma <- sd(Luxembourg_theta$residuals)
Luxembourg_alpha <- Luxembourg_theta$model$alpha
Luxembourg_drift <- Luxembourg_theta$model$drift

#simulate paths manually
Luxembourg_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Luxembourg$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Luxembourg_sigma)
    path[t] <- ell+Luxembourg_drift*t+eps
    ell     <- Luxembourg_alpha*path[t]+(1-Luxembourg_alpha)*ell
  }
  Luxembourg_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Luxembourg", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Luxembourg_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Luxembourg_exports <- exports_forecasts[exports_forecasts$Country == "Luxembourg", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Luxembourg_exports$value),xlab="Year", 
     ylab="exports", main="Luxembourg exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Luxembourg_exports[Luxembourg_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Netherlands----
Netherlands_theta <- thetaf(Netherlands$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Netherlands_sigma <- sd(Netherlands_theta$residuals)
Netherlands_alpha <- Netherlands_theta$model$alpha
Netherlands_drift <- Netherlands_theta$model$drift

#simulate paths manually
Netherlands_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Netherlands$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Netherlands_sigma)
    path[t] <- ell+Netherlands_drift*t+eps
    ell     <- Netherlands_alpha*path[t]+(1-Netherlands_alpha)*ell
  }
  Netherlands_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Netherlands", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Netherlands_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Netherlands_exports <- exports_forecasts[exports_forecasts$Country == "Netherlands", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Netherlands_exports$value),xlab="Year", 
     ylab="exports", main="Netherlands exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Netherlands_exports[Netherlands_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Portugal----
Portugal_theta <- thetaf(Portugal$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Portugal_sigma <- sd(Portugal_theta$residuals)
Portugal_alpha <- Portugal_theta$model$alpha
Portugal_drift <- Portugal_theta$model$drift

#simulate paths manually
Portugal_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Portugal$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Portugal_sigma)
    path[t] <- ell+Portugal_drift*t+eps
    ell     <- Portugal_alpha*path[t]+(1-Portugal_alpha)*ell
  }
  Portugal_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Portugal", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Portugal_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Portugal_exports <- exports_forecasts[exports_forecasts$Country == "Portugal", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Portugal_exports$value),xlab="Year", 
     ylab="exports", main="Portugal exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Portugal_exports[Portugal_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Spain----
Spain_theta <- thetaf(Spain$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Spain_sigma <- sd(Spain_theta$residuals)
Spain_alpha <- Spain_theta$model$alpha
Spain_drift <- Spain_theta$model$drift

#simulate paths manually
Spain_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Spain$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Spain_sigma)
    path[t] <- ell+Spain_drift*t+eps
    ell     <- Spain_alpha*path[t]+(1-Spain_alpha)*ell
  }
  Spain_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Spain", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Spain_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Spain_exports <- exports_forecasts[exports_forecasts$Country == "Spain", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Spain_exports$value),xlab="Year", 
     ylab="exports", main="Spain exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Spain_exports[Spain_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Sweden----
Sweden_theta <- thetaf(Sweden$exports, h=8, level=95)

#extract information from the underlying state space of the theta model
Sweden_sigma <- sd(Sweden_theta$residuals)
Sweden_alpha <- Sweden_theta$model$alpha
Sweden_drift <- Sweden_theta$model$drift

#simulate paths manually
Sweden_exports_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Sweden$exports, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Sweden_sigma)
    path[t] <- ell+Sweden_drift*t+eps
    ell     <- Sweden_alpha*path[t]+(1-Sweden_alpha)*ell
  }
  Sweden_exports_sims[, i] <- path
}

#change to long format and append to dataframe
exports_rows <- data.frame(Country = "Sweden", model = "Theta", variable = "exports", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Sweden_exports_sims))
exports_forecasts <- rbind(exports_forecasts, exports_rows)

#plot forecasts
Sweden_exports <- exports_forecasts[exports_forecasts$Country == "Sweden", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Sweden_exports$value),xlab="Year", 
     ylab="exports", main="Sweden exports - Simulated Paths")
for(i in 1:1000){
  sim_data <- Sweden_exports[Sweden_exports$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#save locally as a csv----
write.csv(exports_forecasts, file='exports_forecasts.csv')
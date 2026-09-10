#packages----
library(readxl)
library(dplyr)
library(Metrics)
library(forecast)

#read in the data----
Austria <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Austria.xlsx")[,c('Year', 'co2')])
Belgium <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Belgium.xlsx")[,c('Year', 'co2')])
Denmark<- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Denmark.xlsx")[,c('Year', 'co2')])
Finland <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Finland.xlsx")[,c('Year', 'co2')])
France <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/France.xlsx")[,c('Year', 'co2')])
Germany <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Germany.xlsx")[,c('Year', 'co2')])
Ireland <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Ireland.xlsx")[,c('Year', 'co2')])
Italy <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Italy.xlsx")[,c('Year', 'co2')])
Luxembourg <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Luxembourg.xlsx")[,c('Year', 'co2')])
Netherlands <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Netherlands.xlsx")[,c('Year', 'co2')])
Portugal <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Portugal.xlsx")[,c('Year', 'co2')])
Spain <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Spain.xlsx")[,c('Year', 'co2')])
Sweden <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Sweden.xlsx")[,c('Year', 'co2')])

#create empty dataframe----
co2_forecasts <- data.frame(country=character(), model=character(), 
                            variable=character(), sim_id=numeric(),
                            year=numeric(),value=numeric())

#Austria----
Austria_co2_log <- log(Austria$co2)
Austria_theta <- thetaf(Austria_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Austria_sigma <- sd(Austria_theta$residuals)
Austria_alpha <- Austria_theta$model$alpha
Austria_drift <- Austria_theta$model$drift

#simulate paths manually
Austria_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Austria_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Austria_sigma)
    path[t] <- ell + Austria_drift * t + eps
    ell     <- Austria_alpha*path[t]+(1-Austria_alpha)*ell
  }
  Austria_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Austria", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Austria_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Austria_co2 <- co2_forecasts[co2_forecasts$Country == "Austria", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Austria_co2$value),xlab="Year", 
     ylab="co2", main="Austria co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Austria_co2[Austria_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Belgium----
Belgium_co2_log <- log(Belgium$co2)
Belgium_theta <- thetaf(Belgium_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Belgium_sigma <- sd(Belgium_theta$residuals)
Belgium_alpha <- Belgium_theta$model$alpha
Belgium_drift <- Belgium_theta$model$drift

#simulate paths manually
Belgium_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Belgium_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Belgium_sigma)
    path[t] <- ell + Belgium_drift * t + eps
    ell     <- Belgium_alpha*path[t]+(1-Belgium_alpha)*ell
  }
  Belgium_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Belgium", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Belgium_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Belgium_co2 <- co2_forecasts[co2_forecasts$Country == "Belgium", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Belgium_co2$value),xlab="Year", 
     ylab="co2", main="Belgium co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Belgium_co2[Belgium_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Denmark----
Denmark_co2_log <- log(Denmark$co2)
Denmark_theta <- thetaf(Denmark_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Denmark_sigma <- sd(Denmark_theta$residuals)
Denmark_alpha <- Denmark_theta$model$alpha
Denmark_drift <- Denmark_theta$model$drift

#simulate paths manually
Denmark_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Denmark_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Denmark_sigma)
    path[t] <- ell + Denmark_drift * t + eps
    ell     <- Denmark_alpha*path[t]+(1-Denmark_alpha)*ell
  }
  Denmark_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Denmark", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Denmark_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Denmark_co2 <- co2_forecasts[co2_forecasts$Country == "Denmark", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Denmark_co2$value),xlab="Year", 
     ylab="co2", main="Denmark co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Denmark_co2[Denmark_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Finland----
Finland_co2_log <- log(Finland$co2)
Finland_theta <- thetaf(Finland_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Finland_sigma <- sd(Finland_theta$residuals)
Finland_alpha <- Finland_theta$model$alpha
Finland_drift <- Finland_theta$model$drift

#simulate paths manually
Finland_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Finland_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Finland_sigma)
    path[t] <- ell + Finland_drift * t + eps
    ell     <- Finland_alpha*path[t]+(1-Finland_alpha)*ell
  }
  Finland_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Finland", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Finland_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Finland_co2 <- co2_forecasts[co2_forecasts$Country == "Finland", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Finland_co2$value),xlab="Year", 
     ylab="co2", main="Finland co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Finland_co2[Finland_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#France----
France_co2_log <- log(France$co2)
France_theta <- thetaf(France_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
France_sigma <- sd(France_theta$residuals)
France_alpha <- France_theta$model$alpha
France_drift <- France_theta$model$drift

#simulate paths manually
France_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(France_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, France_sigma)
    path[t] <- ell + France_drift * t + eps
    ell     <- France_alpha*path[t]+(1-France_alpha)*ell
  }
  France_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "France", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(France_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
France_co2 <- co2_forecasts[co2_forecasts$Country == "France", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(France_co2$value),xlab="Year", 
     ylab="co2", main="France co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- France_co2[France_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Germany----
Germany_co2_log <- log(Germany$co2)
Germany_theta <- thetaf(Germany_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Germany_sigma <- sd(Germany_theta$residuals)
Germany_alpha <- Germany_theta$model$alpha
Germany_drift <- Germany_theta$model$drift

#simulate paths manually
Germany_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Germany_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Germany_sigma)
    path[t] <- ell + Germany_drift * t + eps
    ell     <- Germany_alpha*path[t]+(1-Germany_alpha)*ell
  }
  Germany_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Germany", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Germany_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Germany_co2 <- co2_forecasts[co2_forecasts$Country == "Germany", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Germany_co2$value),xlab="Year", 
     ylab="co2", main="Germany co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Germany_co2[Germany_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Italy----
Italy_co2_log <- log(Italy$co2)
Italy_theta <- thetaf(Italy_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Italy_sigma <- sd(Italy_theta$residuals)
Italy_alpha <- Italy_theta$model$alpha
Italy_drift <- Italy_theta$model$drift

#simulate paths manually
Italy_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Italy_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Italy_sigma)
    path[t] <- ell + Italy_drift * t + eps
    ell     <- Italy_alpha*path[t]+(1-Italy_alpha)*ell
  }
  Italy_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Italy", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Italy_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Italy_co2 <- co2_forecasts[co2_forecasts$Country == "Italy", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Italy_co2$value),xlab="Year", 
     ylab="co2", main="Italy co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Italy_co2[Italy_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Ireland----
Ireland_co2_log <- log(Ireland$co2)
Ireland_theta <- thetaf(Ireland_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Ireland_sigma <- sd(Ireland_theta$residuals)
Ireland_alpha <- Ireland_theta$model$alpha
Ireland_drift <- Ireland_theta$model$drift

#simulate paths manually
Ireland_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Ireland_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Ireland_sigma)
    path[t] <- ell + Ireland_drift * t + eps
    ell     <- Ireland_alpha*path[t]+(1-Ireland_alpha)*ell
  }
  Ireland_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Ireland", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Ireland_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Ireland_co2 <- co2_forecasts[co2_forecasts$Country == "Ireland", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Ireland_co2$value),xlab="Year", 
     ylab="co2", main="Ireland co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Ireland_co2[Ireland_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Luxembourg----
Luxembourg_co2_log <- log(Luxembourg$co2)
Luxembourg_theta <- thetaf(Luxembourg_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Luxembourg_sigma <- sd(Luxembourg_theta$residuals)
Luxembourg_alpha <- Luxembourg_theta$model$alpha
Luxembourg_drift <- Luxembourg_theta$model$drift

#simulate paths manually
Luxembourg_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Luxembourg_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Luxembourg_sigma)
    path[t] <- ell + Luxembourg_drift * t + eps
    ell     <- Luxembourg_alpha*path[t]+(1-Luxembourg_alpha)*ell
  }
  Luxembourg_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Luxembourg", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Luxembourg_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Luxembourg_co2 <- co2_forecasts[co2_forecasts$Country == "Luxembourg", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Luxembourg_co2$value),xlab="Year", 
     ylab="co2", main="Luxembourg co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Luxembourg_co2[Luxembourg_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Netherlands----
Netherlands_co2_log <- log(Netherlands$co2)
Netherlands_theta <- thetaf(Netherlands_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Netherlands_sigma <- sd(Netherlands_theta$residuals)
Netherlands_alpha <- Netherlands_theta$model$alpha
Netherlands_drift <- Netherlands_theta$model$drift

#simulate paths manually
Netherlands_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Netherlands_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Netherlands_sigma)
    path[t] <- ell + Netherlands_drift * t + eps
    ell     <- Netherlands_alpha*path[t]+(1-Netherlands_alpha)*ell
  }
  Netherlands_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Netherlands", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Netherlands_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Netherlands_co2 <- co2_forecasts[co2_forecasts$Country == "Netherlands", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Netherlands_co2$value),xlab="Year", 
     ylab="co2", main="Netherlands co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Netherlands_co2[Netherlands_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Portugal----
Portugal_co2_log <- log(Portugal$co2)
Portugal_theta <- thetaf(Portugal_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Portugal_sigma <- sd(Portugal_theta$residuals)
Portugal_alpha <- Portugal_theta$model$alpha
Portugal_drift <- Portugal_theta$model$drift

#simulate paths manually
Portugal_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Portugal_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Portugal_sigma)
    path[t] <- ell + Portugal_drift * t + eps
    ell     <- Portugal_alpha*path[t]+(1-Portugal_alpha)*ell
  }
  Portugal_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Portugal", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Portugal_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Portugal_co2 <- co2_forecasts[co2_forecasts$Country == "Portugal", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Portugal_co2$value),xlab="Year", 
     ylab="co2", main="Portugal co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Portugal_co2[Portugal_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Spain----
Spain_co2_log <- log(Spain$co2)
Spain_theta <- thetaf(Spain_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Spain_sigma <- sd(Spain_theta$residuals)
Spain_alpha <- Spain_theta$model$alpha
Spain_drift <- Spain_theta$model$drift

#simulate paths manually
Spain_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Spain_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Spain_sigma)
    path[t] <- ell + Spain_drift * t + eps
    ell     <- Spain_alpha*path[t]+(1-Spain_alpha)*ell
  }
  Spain_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Spain", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Spain_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Spain_co2 <- co2_forecasts[co2_forecasts$Country == "Spain", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Spain_co2$value),xlab="Year", 
     ylab="co2", main="Spain co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Spain_co2[Spain_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Sweden----
Sweden_co2_log <- log(Sweden$co2)
Sweden_theta <- thetaf(Sweden_co2_log, h=8, level=95)

#extract information from the underlying state space of the theta model
Sweden_sigma <- sd(Sweden_theta$residuals)
Sweden_alpha <- Sweden_theta$model$alpha
Sweden_drift <- Sweden_theta$model$drift

#simulate paths manually
Sweden_co2_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  ell  <- tail(Sweden_co2_log, 1)
  path <- numeric(8)
  for(t in 1:8){
    eps     <- rnorm(1, 0, Sweden_sigma)
    path[t] <- ell + Sweden_drift * t + eps
    ell     <- Sweden_alpha*path[t]+(1-Sweden_alpha)*ell
  }
  Sweden_co2_sims[, i] <- exp(path)  # back-transform
}

#change to long format and append to dataframe
co2_rows <- data.frame(Country = "Sweden", model = "Theta", variable = "co2", 
                       sim_id=rep(1:1000, each=8), year=rep(2023:2030, times=1000), 
                       value=as.vector(Sweden_co2_sims))
co2_forecasts <- rbind(co2_forecasts, co2_rows)

#plot forecasts
Sweden_co2 <- co2_forecasts[co2_forecasts$Country == "Sweden", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Sweden_co2$value),xlab="Year", 
     ylab="co2", main="Sweden co2 - Simulated Paths")
for(i in 1:1000){
  sim_data <- Sweden_co2[Sweden_co2$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#saving the output locally as a csv----
write.csv(co2_forecasts, file='co2_forecasts.csv')

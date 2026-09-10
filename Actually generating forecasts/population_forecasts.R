#packages----
library(readxl)
library(dplyr)
library(Metrics)
library(forecast)

#read in the data----
Austria <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Austria.xlsx")[,c('Year', 'Population')])
Belgium <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Belgium.xlsx")[,c('Year', 'Population')])
Denmark<- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Denmark.xlsx")[,c('Year', 'Population')])
Finland <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Finland.xlsx")[,c('Year', 'Population')])
France <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/France.xlsx")[,c('Year', 'Population')])
Germany <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Germany.xlsx")[,c('Year', 'Population')])
Ireland <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Ireland.xlsx")[,c('Year', 'Population')])
Italy <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Italy.xlsx")[,c('Year', 'Population')])
Luxembourg <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Luxembourg.xlsx")[,c('Year', 'Population')])
Netherlands <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Netherlands.xlsx")[,c('Year', 'Population')])
Portugal <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Portugal.xlsx")[,c('Year', 'Population')])
Spain <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Spain.xlsx")[,c('Year', 'Population')])
Sweden <- na.omit(read_excel("Desktop/Covariate Forecasting/Country data/Sweden.xlsx")[,c('Year', 'Population')])

#create empty dataframe----
Population_forecasts <- data.frame(country=character(), model=character(), 
                                variable=character(), sim_id=numeric(),
                                year=numeric(),value=numeric())

#Austria----
Austria_theta <- thetaf(Austria$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Austria_sigma <- sd(Austria_theta$residuals)
Austria_alpha <- Austria_theta$model$alpha
Austria_drift <- Austria_theta$model$drift

#simulate paths manually
Austria_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Austria$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Austria_sigma)
    path[t] <- ell+Austria_drift*t+eps
    ell     <- Austria_alpha*path[t]+(1-Austria_alpha)*ell
  }
  Austria_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Austria", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Austria_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Austria_Population <- Population_forecasts[Population_forecasts$Country == "Austria", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Austria_Population$value),xlab="Year", 
     ylab="Population", main="Austria Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Austria_Population[Austria_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Belgium----
Belgium_theta <- thetaf(Belgium$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Belgium_sigma <- sd(Belgium_theta$residuals)
Belgium_alpha <- Belgium_theta$model$alpha
Belgium_drift <- Belgium_theta$model$drift

#simulate paths manually
Belgium_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Belgium$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Belgium_sigma)
    path[t] <- ell+Belgium_drift*t+eps
    ell     <- Belgium_alpha*path[t]+(1-Belgium_alpha)*ell
  }
  Belgium_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Belgium", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Belgium_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Belgium_Population <- Population_forecasts[Population_forecasts$Country == "Belgium", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Belgium_Population$value),xlab="Year", 
     ylab="Population", main="Belgium Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Belgium_Population[Belgium_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Denmark----
Denmark_theta <- thetaf(Denmark$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Denmark_sigma <- sd(Denmark_theta$residuals)
Denmark_alpha <- Denmark_theta$model$alpha
Denmark_drift <- Denmark_theta$model$drift

#simulate paths manually
Denmark_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Denmark$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Denmark_sigma)
    path[t] <- ell+Denmark_drift*t+eps
    ell     <- Denmark_alpha*path[t]+(1-Denmark_alpha)*ell
  }
  Denmark_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Denmark", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Denmark_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Denmark_Population <- Population_forecasts[Population_forecasts$Country == "Denmark", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Denmark_Population$value),xlab="Year", 
     ylab="Population", main="Denmark Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Denmark_Population[Denmark_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Finland----
Finland_theta <- thetaf(Finland$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Finland_sigma <- sd(Finland_theta$residuals)
Finland_alpha <- Finland_theta$model$alpha
Finland_drift <- Finland_theta$model$drift

#simulate paths manually
Finland_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Finland$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Finland_sigma)
    path[t] <- ell+Finland_drift*t+eps
    ell     <- Finland_alpha*path[t]+(1-Finland_alpha)*ell
  }
  Finland_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Finland", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Finland_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Finland_Population <- Population_forecasts[Population_forecasts$Country == "Finland", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Finland_Population$value),xlab="Year", 
     ylab="Population", main="Finland Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Finland_Population[Finland_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#France----
France_theta <- thetaf(France$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
France_sigma <- sd(France_theta$residuals)
France_alpha <- France_theta$model$alpha
France_drift <- France_theta$model$drift

#simulate paths manually
France_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(France$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, France_sigma)
    path[t] <- ell+France_drift*t+eps
    ell     <- France_alpha*path[t]+(1-France_alpha)*ell
  }
  France_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "France", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(France_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
France_Population <- Population_forecasts[Population_forecasts$Country == "France", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(France_Population$value),xlab="Year", 
     ylab="Population", main="France Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- France_Population[France_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Germany----
Germany_theta <- thetaf(Germany$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Germany_sigma <- sd(Germany_theta$residuals)
Germany_alpha <- Germany_theta$model$alpha
Germany_drift <- Germany_theta$model$drift

#simulate paths manually
Germany_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Germany$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Germany_sigma)
    path[t] <- ell+Germany_drift*t+eps
    ell     <- Germany_alpha*path[t]+(1-Germany_alpha)*ell
  }
  Germany_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Germany", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Germany_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Germany_Population <- Population_forecasts[Population_forecasts$Country == "Germany", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Germany_Population$value),xlab="Year", 
     ylab="Population", main="Germany Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Germany_Population[Germany_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Ireland----
Ireland_theta <- thetaf(Ireland$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Ireland_sigma <- sd(Ireland_theta$residuals)
Ireland_alpha <- Ireland_theta$model$alpha
Ireland_drift <- Ireland_theta$model$drift

#simulate paths manually
Ireland_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Ireland$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Ireland_sigma)
    path[t] <- ell+Ireland_drift*t+eps
    ell     <- Ireland_alpha*path[t]+(1-Ireland_alpha)*ell
  }
  Ireland_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Ireland", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Ireland_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Ireland_Population <- Population_forecasts[Population_forecasts$Country == "Ireland", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Ireland_Population$value),xlab="Year", 
     ylab="Population", main="Ireland Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Ireland_Population[Ireland_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Italy----
Italy_theta <- thetaf(Italy$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Italy_sigma <- sd(Italy_theta$residuals)
Italy_alpha <- Italy_theta$model$alpha
Italy_drift <- Italy_theta$model$drift

#simulate paths manually
Italy_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Italy$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Italy_sigma)
    path[t] <- ell+Italy_drift*t+eps
    ell     <- Italy_alpha*path[t]+(1-Italy_alpha)*ell
  }
  Italy_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Italy", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Italy_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Italy_Population <- Population_forecasts[Population_forecasts$Country == "Italy", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Italy_Population$value),xlab="Year", 
     ylab="Population", main="Italy Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Italy_Population[Italy_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Luxembourg----
Luxembourg_theta <- thetaf(Luxembourg$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Luxembourg_sigma <- sd(Luxembourg_theta$residuals)
Luxembourg_alpha <- Luxembourg_theta$model$alpha
Luxembourg_drift <- Luxembourg_theta$model$drift

#simulate paths manually
Luxembourg_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Luxembourg$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Luxembourg_sigma)
    path[t] <- ell+Luxembourg_drift*t+eps
    ell     <- Luxembourg_alpha*path[t]+(1-Luxembourg_alpha)*ell
  }
  Luxembourg_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Luxembourg", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Luxembourg_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Luxembourg_Population <- Population_forecasts[Population_forecasts$Country == "Luxembourg", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Luxembourg_Population$value),xlab="Year", 
     ylab="Population", main="Luxembourg Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Luxembourg_Population[Luxembourg_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Netherlands----
Netherlands_theta <- thetaf(Netherlands$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Netherlands_sigma <- sd(Netherlands_theta$residuals)
Netherlands_alpha <- Netherlands_theta$model$alpha
Netherlands_drift <- Netherlands_theta$model$drift

#simulate paths manually
Netherlands_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Netherlands$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Netherlands_sigma)
    path[t] <- ell+Netherlands_drift*t+eps
    ell     <- Netherlands_alpha*path[t]+(1-Netherlands_alpha)*ell
  }
  Netherlands_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Netherlands", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Netherlands_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Netherlands_Population <- Population_forecasts[Population_forecasts$Country == "Netherlands", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Netherlands_Population$value),xlab="Year", 
     ylab="Population", main="Netherlands Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Netherlands_Population[Netherlands_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Portugal----
Portugal_theta <- thetaf(Portugal$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Portugal_sigma <- sd(Portugal_theta$residuals)
Portugal_alpha <- Portugal_theta$model$alpha
Portugal_drift <- Portugal_theta$model$drift

#simulate paths manually
Portugal_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Portugal$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Portugal_sigma)
    path[t] <- ell+Portugal_drift*t+eps
    ell     <- Portugal_alpha*path[t]+(1-Portugal_alpha)*ell
  }
  Portugal_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Portugal", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Portugal_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Portugal_Population <- Population_forecasts[Population_forecasts$Country == "Portugal", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Portugal_Population$value),xlab="Year", 
     ylab="Population", main="Portugal Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Portugal_Population[Portugal_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Spain----
Spain_theta <- thetaf(Spain$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Spain_sigma <- sd(Spain_theta$residuals)
Spain_alpha <- Spain_theta$model$alpha
Spain_drift <- Spain_theta$model$drift

#simulate paths manually
Spain_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Spain$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Spain_sigma)
    path[t] <- ell+Spain_drift*t+eps
    ell     <- Spain_alpha*path[t]+(1-Spain_alpha)*ell
  }
  Spain_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Spain", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Spain_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Spain_Population <- Population_forecasts[Population_forecasts$Country == "Spain", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Spain_Population$value),xlab="Year", 
     ylab="Population", main="Spain Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Spain_Population[Spain_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Sweden----
Sweden_theta <- thetaf(Sweden$Population, h=7, level=95)

#extract information from the underlying state space of the theta model
Sweden_sigma <- sd(Sweden_theta$residuals)
Sweden_alpha <- Sweden_theta$model$alpha
Sweden_drift <- Sweden_theta$model$drift

#simulate paths manually
Sweden_Population_sims <- matrix(NA, nrow=7, ncol=1000)
set.seed(20229797)
for(i in 1:1000){
  ell  <- tail(Sweden$Population, 1)
  path <- numeric(7)
  for(t in 1:7){
    eps     <- rnorm(1, 0, Sweden_sigma)
    path[t] <- ell+Sweden_drift*t+eps
    ell     <- Sweden_alpha*path[t]+(1-Sweden_alpha)*ell
  }
  Sweden_Population_sims[, i] <- path
}

#change to long format and append to dataframe
Population_rows <- data.frame(Country = "Sweden", model = "Theta", variable = "Population", 
                           sim_id=rep(1:1000, each=7), year=rep(2024:2030, times=1000), 
                           value=as.vector(Sweden_Population_sims))
Population_forecasts <- rbind(Population_forecasts, Population_rows)

#plot forecasts
Sweden_Population <- Population_forecasts[Population_forecasts$Country == "Sweden", ]
plot(NULL, xlim=c(2024, 2030), ylim=range(Sweden_Population$value),xlab="Year", 
     ylab="Population", main="Sweden Population - Simulated Paths")
for(i in 1:1000){
  sim_data <- Sweden_Population[Sweden_Population$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#save locally as a csv----
write.csv(Population_forecasts, file='new_Population_forecasts.csv')

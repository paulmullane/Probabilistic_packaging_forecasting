#packages----
library(readxl)
library(dplyr)
library(Metrics)
library(forecast)

#read in the data----
Austria <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Austria.xlsx")[,c('Year', 'material_footprint')])
Belgium <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Belgium.xlsx")[,c('Year', 'material_footprint')])
Denmark<- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Denmark.xlsx")[,c('Year', 'material_footprint')])
Finland <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Finland.xlsx")[,c('Year', 'material_footprint')])
France <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/France.xlsx")[,c('Year', 'material_footprint')])
Germany <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Germany.xlsx")[,c('Year', 'material_footprint')])
Ireland <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Ireland.xlsx")[,c('Year', 'material_footprint')])
Italy <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Italy.xlsx")[,c('Year', 'material_footprint')])
Luxembourg <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Luxembourg.xlsx")[,c('Year', 'material_footprint')])
Netherlands <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Netherlands.xlsx")[,c('Year', 'material_footprint')])
Portugal <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Portugal.xlsx")[,c('Year', 'material_footprint')])
Spain <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Spain.xlsx")[,c('Year', 'material_footprint')])
Sweden <- na.omit(read_excel("C:/Users/20229798/OneDrive - University of Limerick/Desktop/Covariate Forecasting/Country data/Sweden.xlsx")[,c('Year', 'material_footprint')])

#create empty dataframe----
material_footprint_forecasts <- data.frame(ountry=character(), model=character(), 
                                           variable=character(), sim_id=numeric(),
                                           year=numeric(),value=numeric())

#Austria----
austria_arima <- auto.arima(Austria$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
austria_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  austria_mf_sims[, i] <- simulate(austria_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Austria", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(austria_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
austria_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Austria", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(austria_mf$value),
     xlab="Year", ylab="Material Footprint", main="Austria Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- austria_mf[austria_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Belgium----
Belgium_arima <- auto.arima(Belgium$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Belgium_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Belgium_mf_sims[, i] <- simulate(Belgium_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Belgium", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Belgium_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Belgium_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Belgium", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Belgium_mf$value),
     xlab="Year", ylab="Material Footprint", main="Belgium Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Belgium_mf[Belgium_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}


#Denmark----
Denmark_arima <- auto.arima(Denmark$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Denmark_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Denmark_mf_sims[, i] <- simulate(Denmark_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Denmark", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Denmark_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Denmark_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Denmark", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Denmark_mf$value),
     xlab="Year", ylab="Material Footprint", main="Denmark Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Denmark_mf[Denmark_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Finland----
Finland_arima <- auto.arima(Finland$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Finland_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Finland_mf_sims[, i] <- simulate(Finland_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Finland", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Finland_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Finland_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Finland", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Finland_mf$value),
     xlab="Year", ylab="Material Footprint", main="Finland Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Finland_mf[Finland_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#France----
France_arima <- auto.arima(France$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
France_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  France_mf_sims[, i] <- simulate(France_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "France", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(France_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
France_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "France", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(France_mf$value),
     xlab="Year", ylab="Material Footprint", main="France Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- France_mf[France_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Germany----
Germany_arima <- auto.arima(Germany$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Germany_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Germany_mf_sims[, i] <- simulate(Germany_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Germany", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Germany_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Germany_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Germany", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Germany_mf$value),
     xlab="Year", ylab="Material Footprint", main="Germany Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Germany_mf[Germany_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Ireland----
Ireland_arima <- auto.arima(Ireland$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Ireland_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Ireland_mf_sims[, i] <- simulate(Ireland_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Ireland", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Ireland_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Ireland_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Ireland", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Ireland_mf$value),
     xlab="Year", ylab="Material Footprint", main="Ireland Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Ireland_mf[Ireland_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Italy----
Italy_arima <- auto.arima(Italy$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Italy_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Italy_mf_sims[, i] <- simulate(Italy_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Italy", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Italy_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Italy_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Italy", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Italy_mf$value),
     xlab="Year", ylab="Material Footprint", main="Italy Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Italy_mf[Italy_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Luxembourg----
Luxembourg_arima <- auto.arima(Luxembourg$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Luxembourg_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Luxembourg_mf_sims[, i] <- simulate(Luxembourg_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Luxembourg", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Luxembourg_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Luxembourg_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Luxembourg", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Luxembourg_mf$value),
     xlab="Year", ylab="Material Footprint", main="Luxembourg Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Luxembourg_mf[Luxembourg_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Netherlands----
Netherlands_arima <- auto.arima(Netherlands$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Netherlands_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Netherlands_mf_sims[, i] <- simulate(Netherlands_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Netherlands", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Netherlands_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Netherlands_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Netherlands", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Netherlands_mf$value),
     xlab="Year", ylab="Material Footprint", main="Netherlands Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Netherlands_mf[Netherlands_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Portugal----
Portugal_arima <- auto.arima(Portugal$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Portugal_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Portugal_mf_sims[, i] <- simulate(Portugal_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Portugal", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Portugal_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Portugal_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Portugal", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Portugal_mf$value),
     xlab="Year", ylab="Material Footprint", main="Portugal Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Portugal_mf[Portugal_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Spain----
Spain_arima <- auto.arima(Spain$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Spain_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Spain_mf_sims[, i] <- simulate(Spain_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Spain", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Spain_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Spain_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Spain", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Spain_mf$value),
     xlab="Year", ylab="Material Footprint", main="Spain Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Spain_mf[Spain_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#Sweden----
Sweden_arima <- auto.arima(Sweden$material_footprint, seasonal=FALSE, max.p=7, 
                            max.q=7)

#create matrix to store simulation results
Sweden_mf_sims <- matrix(NA, nrow=8, ncol=1000)
set.seed(20229798)
for(i in 1:1000){
  Sweden_mf_sims[, i] <- simulate(Sweden_arima, nsim=8, future=TRUE)
}

#change to long format and append to dataframe
mf_rows <- data.frame(country = "Sweden", model = "ARIMA", 
                      variable = "material_footprint", sim_id=rep(1:1000, each=8),
                      year=rep(2023:2030, times=1000), value=as.vector(Sweden_mf_sims))
material_footprint_forecasts <- rbind(material_footprint_forecasts, mf_rows)

#plot forecasted paths
Sweden_mf <- material_footprint_forecasts[material_footprint_forecasts$country == "Sweden", ]
plot(NULL, xlim=c(2023, 2030), ylim=range(Sweden_mf$value),
     xlab="Year", ylab="Material Footprint", main="Sweden Material Footprint - Simulated Paths")
for(i in 1:1000){
  sim_data <- Sweden_mf[Sweden_mf$sim_id == i, ]
  lines(sim_data$year, sim_data$value, col=rgb(0,0,1,0.05))
}

#saving the output locally as csv----
write.csv(material_footprint_forecasts, file='material_footprint_forecasts.csv')

#loading packages----
library(readr)
library(ggplot2)
library(cowplot)
theme_set(theme_cowplot())

#reading data----
meta_data <- read_csv('Desktop/Packaging waste forecasting/Meta Modelling/Creating Meta Datasets/meta_dataset.csv')

#select numerical variables from the larger dataset----
pca_data <- meta_data[, c('n_covariates', 'train_sd', 'test_sd', 'sd_ratio', 
                          'slope_full', 'slope_train', 'slope_diff', 
                          'max_abs_change', 'mean_abs_change')]

#run PCA (this also scales variables which is needed for PCA)----
pca_result <- prcomp(pca_data, scale.=TRUE, center=TRUE)
summary(pca_result)

#extract scores----
#here we are getting the PC scores for each row in the meta dataset
pca_scores <- as.data.frame(pca_result$x)
pca_scores$country <- meta_data$country
pca_scores$stream  <- meta_data$stream
pca_scores$model   <- meta_data$model

#plot of first 2 PCs coloured by waste stream----
ggplot(pca_scores, aes(x=PC1, y=PC2, colour=stream, label=country)) +
  geom_point(size=3) +
  geom_text(vjust=-0.5, size=3) +
  theme_minimal() +
  labs(title="PCA of Meta-Dataset — coloured by stream")

#plot of 2 pcs by forecast error----
ggplot(pca_scores, aes(x=PC1, y=PC2, 
                       colour=meta_data$abs_forecast_error_2023,
                       label=country)) +
  geom_point(size=3) +
  geom_text(vjust=-0.5, size=3) +
  scale_colour_gradient(low="yellow", high="black") +
  labs(title="PCA — coloured by absolute forecast error 2023")

#plot of 2 PCs coloured by PI width----
ggplot(pca_scores, aes(x=PC1, y=PC2,
                       colour=meta_data$pi_width_2023,
                       label=country)) +
  geom_point(size=3) +
  geom_text(vjust=-0.5, size=3) +
  scale_colour_gradient(low="green", high="black") +
  labs(title="PCA — coloured by PI width 2023")


#plot of 2 PCs coloured by PI width & shaped by strean----
ggplot(pca_scores, aes(x=PC1, y=PC2,
                       colour=meta_data$pi_width_2023,
                       shape=meta_data$stream,
                       label=country)) +
  geom_point(size=3) +
  geom_text(vjust=-0.5, size=3) +
  scale_colour_gradient(low="green", high="black") +
  scale_shape_manual(values=c(16, 17, 15, 18), labels = c("Glass", "Metallic", "Paper & Cardboard", "Plastic")) +
  labs(colour="PI Width 2023", shape="Stream")

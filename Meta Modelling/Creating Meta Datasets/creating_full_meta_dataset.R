library(readr)
#read in the data----
glass_meta_dataset <- read_csv("glass_meta_dataset.csv")
plastic_meta_dataset <- read_csv("plastic_meta_dataset.csv")
metallic_meta_dataset <- read_csv("metallic_meta_dataset.csv")
paper_cardboard_meta_dataset <- read_csv("paper_cardboard_meta_dataset.csv")

#combining datasets----
full_meta_data <- rbind(glass_meta_dataset, plastic_meta_dataset, 
                        metallic_meta_dataset, paper_cardboard_meta_dataset,)

#write the dataset locally----
write.csv(full_meta_data[, -1], file='meta_dataset.csv') #the -1 removes the counter variable

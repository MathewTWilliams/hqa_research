# Author: Matt Williams
# Version: 2/13/2023

data <- read.csv("persistence_entropies.csv", header = TRUE)

recon_names <- c("data_original", "data_jpg", "data_recon_0", "data_recon_1", "data_recon_2", "data_recon_3", "data_recon_4")

attach(data)
library(aplpack)

for (label in unique(data$Label)) {
  for (recon in recon_names) {
      
    recon_df <- data[data$Reconstruction == recon & data$Label == label,]
    
    h0_min <- min(recon_df$H0, na.rm = FALSE) - 0.25
    h0_max <- max(recon_df$H0, na.rm = FALSE) + 0.25
    h1_min <- min(recon_df$H1, na.rm = FALSE) - 0.25
    h1_max <- max(recon_df$H1, na.rm = FALSE) + 0.25
    
    recon_reg_df <- recon_df[recon_df["Attack"] == "None",]
    recon_atk_df <- recon_df[recon_df["Attack"] != "None",]
    
    recon_reg_img_df <- recon_reg_df[recon_reg_df$Input == "Image",]
    recon_reg_cnn_df <- recon_reg_df[recon_reg_df$Input == "CNN Output",]
    recon_atk_img_df <- recon_atk_df[recon_atk_df$Input == "Image",]
    recon_atk_cnn_df <- recon_atk_df[recon_atk_df$Input == "CNN Output",]
    
    cor_recon_reg_img_df <- subset(recon_reg_img_df, Prediction == Label)
    cor_recon_reg_cnn_df <- subset(recon_reg_cnn_df, Prediction == Label)
    inc_recon_reg_img_df <- subset(recon_reg_img_df, Prediction != Label)
    inc_recon_reg_cnn_df <- subset(recon_reg_cnn_df, Prediction != Label)
    
    cor_recon_atk_img_df <- subset(recon_atk_img_df, Prediction == Label)
    cor_recon_atk_cnn_df <- subset(recon_atk_cnn_df, Prediction == Label)
    inc_recon_atk_img_df <- subset(recon_atk_img_df, Prediction != Label)
    inc_recon_atk_cnn_df <- subset(recon_atk_cnn_df, Prediction != Label)
    
    file_name <- sprintf("./Visuals/Entropies/%s_%d.png", recon, label) 
    png(file_name, width = 1000, height = 1000)
    
    par(mfrow=c(2, 4))
    
    bagplot(cor_recon_reg_img_df[,c("H0", "H1")], factor = 3.5, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE, 
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1",
            main = "Classified Images")
  
    bagplot(cor_recon_reg_cnn_df[,c("H0", "H1")], factor = 3.5, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1",
            main = "Classified CNN Outputs")
    
    bagplot(cor_recon_atk_img_df[,c("H0", "H1")], factor = 3.5, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1",
            main = "Classified Attacked Images")
    
    bagplot(cor_recon_atk_cnn_df[,c("H0", "H1")], factor = 3.5, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1",
            main = "Classified Attacked CNN Ouputs")
    
    bagplot(inc_recon_reg_img_df[,c("H0", "H1")], factor = 3.5, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1",
            main = "Misclassified Images")
    
    bagplot(inc_recon_reg_cnn_df[,c("H0", "H1")], factor = 3.5, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1",
            main = "Misclassified CNN Outputs")


    bagplot(inc_recon_atk_img_df[,c("H0", "H1")], factor = 3.5, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1",
            main = "Misclassified Attacked Images")

    bagplot(inc_recon_atk_cnn_df[,c("H0", "H1")], factor = 3.5, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1",
            main = "Misclassified Attacked CNN Outputs")
    
    dev.off()
  }
}


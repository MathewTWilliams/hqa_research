# Author: Matt Williams
# Version: 6/27/2023

# This file is used to create bag plot visualizations based off the H0 and H1 persistent entropy values.
# A visualization is made for each label, for each reconstruction type.
# Each visualization contains 8 bag plots.
# 1. Entropies from correctly predicted normal images of the current label.
# 2. Entropies from incorrectly predicted normal images misclassified as the current label.
# 3. Entropies from the output of the final Conv2D in the Lenet Classifier (based on correctly predicted normal images of the current label).
# 4. Entropies from the output of the final Conv2D in the Lenet Classifier (based on incorrectly predicted normal images misclassified as the current label).
# 5. Entropies from correctly predicted attacked images of the current label.
# 6. Entropies from incorrectly predicted attacked images misclassified as the current label.
# 7. Entropies from the output of the final Conv2D in the Lenet Classifier (based on correctly predicted attacked images of the current label).
# 8. Entropies from the output of the final Conv2D in the Lenet Classifier (based on incorrectly predicted attacked images misclassified as the current label).
# This file assumes you are using RStudio as your IDE.


library(aplpack)
library(rstudioapi)

cur_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(cur_path))
getwd()


data <- read.csv("./csv_files/persistence_entropies.csv", header = TRUE)
attach(data)

recon_names <- c("data_original", "data_jpg", "data_recon_0", "data_recon_1", "data_recon_2", "data_recon_3", "data_recon_4")

for (label in unique(data$Label)) {
  for (recon in recon_names) {
      
    recon_df <- data[data$Reconstruction == recon & (data$Label == label | data$Prediction == label),]
    
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
    cor_recon_atk_img_df <- subset(recon_atk_img_df, Prediction == Label)
    cor_recon_atk_cnn_df <- subset(recon_atk_cnn_df, Prediction == Label)
    
    # where Prediction is equal to the current label, but the Prediction and Label columns don't match
    inc_recon_reg_img_df <- subset(recon_reg_img_df, Prediction == label & Prediction != Label)
    inc_recon_reg_cnn_df <- subset(recon_reg_cnn_df, Prediction == label & Prediction != Label)
    inc_recon_atk_img_df <- subset(recon_atk_img_df, Prediction == label & Prediction != Label)
    inc_recon_atk_cnn_df <- subset(recon_atk_cnn_df, Prediction == label & Prediction != Label)
    
    file_name <- sprintf("./Visuals/Entropies/%s_%d.png", recon, label) 
    png(file_name, width = 1200, height = 1200)
    
    par(mfrow=c(2, 4))
    
    bagplot(cor_recon_reg_img_df[,c("H0", "H1")], factor = 3, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE, 
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1", approx.limit = 2000,
            main = sprintf("Classified images of %d from %s: %d", label, recon, nrow(cor_recon_reg_img_df)),
            cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    
    bagplot(inc_recon_reg_img_df[,c("H0", "H1")], factor = 3, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1", approx.limit = 2000,
            main = sprintf("Misclassified Images as %d from %s: %d", label, recon, nrow(inc_recon_reg_img_df)),
            cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    
    bagplot(cor_recon_reg_cnn_df[,c("H0", "H1")], factor = 3, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1", approx.limit = 2000,
            main = sprintf("Classified CNN Outputs of %d from %s: %d", label, recon, nrow(cor_recon_reg_cnn_df)),
            cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    
    bagplot(inc_recon_reg_cnn_df[,c("H0", "H1")], factor = 3, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1", approx.limit = 2000,
            main = sprintf("Misclassified CNN Outputs as %d from %s: %d", label, recon, nrow(inc_recon_reg_cnn_df)),
            cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    
    bagplot(cor_recon_atk_img_df[,c("H0", "H1")], factor = 3, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1", approx.limit = 2000,
            main = sprintf("Classified Attacked Images of %d from %s: %d", label, recon, nrow(cor_recon_atk_img_df)),
            cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)

    bagplot(inc_recon_atk_img_df[,c("H0", "H1")], factor = 3, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4), approx.limit = 2000,
            xlab = "H0", ylab = "H1",
            main = sprintf("Misclassified Attacked Images as %d from %s: %d", label, recon, nrow(inc_recon_atk_img_df)),
            cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    
    bagplot(cor_recon_atk_cnn_df[,c("H0", "H1")], factor = 3, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1", approx.limit = 2000,
            main = sprintf("Classified Attacked CNN Ouputs of %d from %s: %d", label, recon, nrow(cor_recon_atk_cnn_df)),
            cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    
    bagplot(inc_recon_atk_cnn_df[,c("H0", "H1")], factor = 3, create.plot = TRUE,
            show.outlier = TRUE, show.looppoints=TRUE,
            show.bagpoints = TRUE, dkmethod=2,
            show.whiskers=FALSE, show.loophull=TRUE,
            show.baghull=TRUE, verbose = FALSE,
            xlim = c(0,4), ylim = c(0,4),
            xlab = "H0", ylab = "H1", approx.limit = 2000,
            main = sprintf("Misclassified Attacked CNN Outputs as %d from %s: %d", label, recon, nrow(inc_recon_atk_cnn_df)),
            cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    
    dev.off()
  }
}


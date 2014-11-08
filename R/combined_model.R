setwd('C:\\Users\\Ivan.Liuyanfeng\\Desktop\\Data_Mining_Work_Space\\Tradeshift-Text-Classification\\preds')
require(data.table)
data_7 <- data.frame(fread('submission_D22_L011_y33_y6.csv')) # 16897537
data_7_2 <- data.frame(fread('submission_D22_L011_y33.csv')) # 17442624
data_3 <- data.frame(fread('quick_start.csv')) # 17987706

dim(data_7)
dim(data_3)
head(data_7)
head(data_3)

a1 <- 17987706-17442624
a2 <- 17442624-16897537
17987706/a1
17987706/a2

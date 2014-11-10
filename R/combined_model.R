setwd('C:\\Users\\Ivan.Liuyanfeng\\Desktop\\Data_Mining_Work_Space\\Tradeshift-Text-Classification\\preds')
require(data.table)
data_7 <- read.csv('submission_D22_L011_y33_y6_y12_y7.csv')
# data_7 <- data.frame(fread('submission_D22_L011_y33_y6.csv')) # 16897537
# data_7_2 <- data.frame(fread('submission_D22_L011_y33.csv')) # 17442624
data_3 <- data.frame(fread('quick_start.csv')) # 17987706

# dim(data_7)
# dim(data_3)
# head(data_7, 50)
# head(data_3, 50)

# a1 <- 17987706-17442624
# a2 <- 17442624-16897537
# 17987706/a1
# 17987706/a2

# identical(data_7[,1],data_3[,1])
# identical(head(data_7, 50),head(data_3, 50))
# merge(data_7,data_3,by.x = data_7[,1],by.y=data_3[,1])

combined_pred <- 0.5 * data_7[,2] + 0.5 * data_3[,2]
cbind(head(combined_pred), head(data_7[,2]), head(data_3[,2]))
new_pred <- cbind.data.frame(id_label = data_7[,1], pred = combined_pred)
head(new_pred, 50)

write.table(new_pred, file='combined_pred_64_y33_6_12_7.csv',sep = ',', row.names = F)

#########################
### mannual adjustment ###
#########################
y33 <- new_pred[grep('y33',new_pred$id_label),]
dim(y33)
dim(new_pred)

y33_0.99 <- y33[which(y33$pred >= 0.995),]
dim(y33_0.99)

y33_0.99_id <- c()
for (i in 1:length(y33_0.99$id_label)){
    y33_0.99_id <- c(y33_0.99_id, strsplit(as.character(y33_0.99$id_label[i]),'_')[[1]][1])
}

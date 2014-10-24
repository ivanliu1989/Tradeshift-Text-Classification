setwd('C:\\Users\\Ivan.Liuyanfeng\\Desktop\\Data_Mining_Work_Space\\Tradeshift-Text-Classification\\')
setwd('/Users/ivan/Work_directory/TTC')
library(data.table)
rm(list=ls(all=TRUE));gc(reset=TRUE);par(mfrow=c(1,1))

## Big Data -- Sample Data!
train <- as.data.frame(fread("data/train.csv"))
dim(train)
sample_size = 170000
ratio = ncol(train) / sample_size
train_sample=sample(train, ratio)
dim(train_sample)
write.csv(train_sample, 'data/train_sample.csv')

## Try to make something useful
train_sample <- as.data.frame(fread("train_sample.csv"))
labels <- as.data.frame(fread("trainLabels.csv"))
dim(labels)
train_with_labels = merge(train_sample, labels, by = 'id')
dim(train_with_labels)
test <- as.data.frame(fread("test.csv"))

# Categorical values encoding
mean(is.na(train_with_labels[,3]))
train_with_labels[train_with_labels=='YES'] <- 1
train_with_labels[train_with_labels=='NO'] <- 0
train_with_labels[train_with_labels==''] <- NaN
table(train_with_labels[,2])
test[test=='YES'] <- 1
test[test=='NO'] <- 0
test[test==''] <- NaN
table(test[,3])

for(name in names(train_with_labels)){
            if(substr(name, 1, 1)=="x"){
                column_type <- max(table(apply(train_with_labels[name],1, class))
                # LOL expression
                if(column_type){

                	}else{
                		
                	}
            }
        }
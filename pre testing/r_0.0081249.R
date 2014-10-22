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
train_sample <- as.data.frame(fread("data/train_sample.csv"))
labels <- as.data.frame(fread("data/trainLabels.csv"))
dim(labels)
train_with_labels = merge(train_sample, labels, by = 'id')
dim(train_with_labels)
test <- as.data.frame(fread("data/test.csv"))

# Categorical values encoding
train_with_labels[train_with_labels=='Yes'] <- 1
train_with_labels[train_with_labels=='No'] <- 0
train_with_labels[train_with_labels=='nan'] <- NA
test[test=='Yes'] <- 1
test[test=='No'] <- 0
test[test=='nan'] <- NA

for(p in names(train_with_labels)){
            if(substr(p, 1, 1)=="x"){
                column_type
                # LOL expression
                if(column_type){

                	}else{
                		
                	}
            }
        }
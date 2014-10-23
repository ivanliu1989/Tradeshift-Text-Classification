#######################
## Environment Setup ##
#######################
setwd('/Users/ivan/Work_directory/TTC')
setwd('C:\\Users\\Ivan.Liuyanfeng\\Desktop\\Data_Mining_Work_Space\\TTC')
setwd('H:\\Machine Learning\\TTC')
rm(list=ls(all=TRUE));gc(reset=TRUE);par(mfrow=c(1,1))
require(data.table); require(caret); require(reshape)

###############
## Load Data ##
###############
train <- as.data.frame(fread('data/train.csv'))
test <- as.data.frame(fread('data/test.csv'))
labels <- as.data.frame(fread('data/trainLabels.csv'))
dim(train); dim(test); dim(labels)
save(train,test,labels, file='data/raw_data.RData')
# load('data/raw_data.RData')

#################
## Subset Data ##
#################
train_sample_index <- createDataPartition(train$id,p=0.1,list=F)
train_sample <- train[train_sample_index,]
dim(train_sample)
train_sample_labels <- merge(x=train_sample,y=labels,by.x="id", by.y="id")
names(train_sample_labels)

#################################
## Categorical values encoding ##
#################################
train_sample_labels[train_sample_labels=='YES'] <- 1
train_sample_labels[train_sample_labels=='NO'] <- 0
train_sample_labels[train_sample_labels=='nan'] <- NA
test[test=='YES'] <- 1
test[test=='NO'] <- 0
test[test=='nan'] <- NA
head(train_sample_labels); head(test)
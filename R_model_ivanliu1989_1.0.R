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
trainLabels <- as.data.frame(fread('data/trainLabels.csv'))
dim(train); dim(test); dim(trainLabels)
save(train,test,trainLabels, file='data/raw_data.RData')
# load('data/raw_data.RData')

#################
## Subset Data ##
#################
train_sample_index <- createDataPartition(train$id,p=0.1,list=F)
train_sample <- train[train_sample_index,]
dim(train_sample)
# train <- train_sample
train_sample_labels <- merge(x=train_sample,y=trainLabels,by.x="id", by.y="id")
names(train_sample_labels)
table(substr(names(train_sample_labels), 1, 1))

#################################
## Categorical values encoding ##
#################################
# Group variables
vars <- colnames(train)
ID <- vars[1]
labels <- colnames(trainLabels)
predictors <- vars[c(-1,-92)] #remove ID and one features with too many factors
targets <- labels[-1] #remove ID
# Attach the labels to the training data
train_labels <- merge(x=train, y=trainLabels, by.x='id', by.y='id')
rm(train_sample)
rm(train_sample_index)
# Split the training data into train/valid (95%/5%)
# Want to keep train large enough to make a good submission if submitwithfulldata = F
train_labels_index <- createDataPartition(train_labels$id,p=0.95,list=F)
train <- train_labels[train_labels_index,]
valid <- train_labels[-train_labels_index,]
dim(train); dim(valid)







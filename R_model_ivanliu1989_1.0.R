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
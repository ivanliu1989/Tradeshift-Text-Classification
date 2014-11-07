setwd('H:\\Machine Learning\\Tradeshift-Text-Classification')
rm(list=ls(all=TRUE));gc(reset=TRUE);par(mfrow=c(1,1))
load('Data/labels.RData')
load('Data/train.RData')
load('Data/test.RData')

head(labels)
head(train) # 1700000 146
head(test) # 545082 146

train[train=='YES'] <- 1
train[train=='NO'] <- 0
train[train==''] <- NaN
test[test=='YES'] <- 1
test[test=='NO'] <- 0
test[test==''] <- NaN

Total <- rbind(train,test) # 2245082 146
dim(train);dim(test);dim(Total)

head(Total)
dValue <- c()
for (i in names(Total)){
    dValue <- c(dValue, length(table(Total[i])))
}
names(dValue) <- names(Total)
dClass <- c()
for (i in 1:length(names(Total))){
    dClass <- c(dClass, class(Total[,i]))
}
dSum <- rbind(head(Total),dClass,dValue)
write.csv(dSum, 'Feature_engineering.csv')

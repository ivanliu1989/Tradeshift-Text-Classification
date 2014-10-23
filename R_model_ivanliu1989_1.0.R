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
# Settings (at least one of the following two settings has to be TRUE)
validate = T #whether to compute CV error on train/validation split (or n-fold), potentially with grid search
submitwithfulldata = T #whether to use full training dataset for submission (if FALSE, then the validation model(s) will make test set predictions)
ensemble_size <- 2 # more -> lower variance
seed0 = 1337
reproducible_mode = T # Set to TRUE if you want reproducible results, e.g. for final Kaggle submission if you think you'll win :)  Note: will be slower for DL

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
rm(train_sample);rm(train_sample_index)
# Split the training data into train/valid (95%/5%)
# Want to keep train large enough to make a good submission if submitwithfulldata = F
train_labels_index <- createDataPartition(train_labels$id,p=0.95,list=F)
train <- train_labels[train_labels_index,]
valid <- train_labels[-train_labels_index,]
dim(train); dim(valid)
rm(train_labels); rm(train_labels_index); rm(trainLabels);gc(reset=TRUE)

## Main loop over targets
for (resp in 1:length(targets)){
    # always just predict class 0 for y_14 (is constant)
    if (resp == 14) {
        final_submission <- cbind(final_submission, as.data.frame(matrix(0, nrow = nrow(test), ncol = 1)))
        colnames(final_submission)[resp] <- targets[resp]
        next
    }
    if (validate) {
        cat("\n\nNow training and validating an ensemble model for", targets[resp], "...\n")
        train_resp <- train[,targets[resp]]
        valid_resp <- valid[,targets[resp]]
        
        for (n in 1:ensemble_size) {
            cat("\n\nBuilding ensemble validation model", n, "of", ensemble_size, "for", targets[resp], "...\n")
            
            cvmodel <- train(x = predictors, y = targets[resp], data = train, 
                             method = 'rf', seed = seed0 + resp*ensemble_size + n)
            # use probabilities - clamp validation predictions for LogLoss computation
            train_preds <- predict(cvmodel, train)
            valid_preds <- predict(cvmodel, valid) 
            
            # compute LogLoss for this ensemble member, on training data
            cat("\nLogLoss of this ensemble member on training data:", trainLL)
            
            # compute LogLoss for this ensemble member, on validation data
            cat("\nLogLoss of this ensemble member on validation data:", validLL)
            
            if (!submitwithfulldata) {
                test_preds <- predict(cvmodel, test)
            }
            if (n == 1) {
                valid_preds_ensemble <- valid_preds
                train_preds_ensemble <- train_preds
                if (!submitwithfulldata) {
                    test_preds_ensemble <- test_preds
                }
            } else {
                valid_preds_ensemble <- valid_preds_ensemble + valid_preds
                train_preds_ensemble <- train_preds_ensemble + train_preds
                if (!submitwithfulldata) {
                    test_preds_ensemble <- test_preds_ensemble + test_preds
                }
            }
        }
        train_preds <- train_preds_ensemble/ensemble_size ##ensemble average of probabilities
        valid_preds <- valid_preds_ensemble/ensemble_size ##ensemble average of probabilities
        if (!submitwithfulldata) {
            test_preds  <- test_preds_ensemble/ensemble_size
        }
        
        ## Compute LogLoss of ensemble
        cat("\nLogLosses of ensemble on training data so far:", tLogLoss)
        cat("\nMean LogLoss of ensemble on training data so far:", sum(tLogLoss)/resp)
        
        cat("\nLogLosses of ensemble on validation data so far:", vLogLoss)
        cat("\nMean LogLoss of ensemble on validation data so far:", sum(vLogLoss)/resp)
        
        if (!submitwithfulldata) {
            cat("\nMaking test set predictions with ensemble model on 95% of the data\n")
            ensemble_average <- as.data.frame(test_preds) #bring ensemble average to R
            colnames(ensemble_average)[1] <- targets[resp] #give it the right name
            if (resp == 1) {
                final_submission <- ensemble_average
            } else {
                final_submission <- cbind(final_submission, ensemble_average)
            }
            #print(head(final_submission))
        }
    }
    
    if (validate) {
        cat("\nOverall training LogLosses = " , tLogLoss)
        cat("\nOverall training LogLoss = " , mean(tLogLoss))
        cat("\nOverall validation LogLosses = " , vLogLoss)
        cat("\nOverall validation LogLoss = " , mean(vLogLoss))
        cat("\n")
    }
    
    print(summary(final_submission))
    submission <- read.csv(path_submission)
    #reshape predictions into 1D
    fs <- t(as.matrix(final_submission))
    dim(fs) <- c(prod(dim(fs)),1)
    submission[,2] <- fs #replace 0s with actual predictions
    write.csv(submission, file = "./submission.csv", quote = F, row.names = F)
    sink()
}






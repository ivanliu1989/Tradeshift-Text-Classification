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
            
            cvmodel <- train(x=train[,predictors], y=train_resp, data = train, 
                             method = 'rf', seed = seed0 + resp*ensemble_size + n)
            # use probabilities - clamp validation predictions for LogLoss computation
            train_preds <- predict(cvmodel, train)
            valid_preds <- predict(cvmodel, valid) 
            
            # compute LogLoss for this ensemble member, on training data
            cat("\nLogLoss of this ensemble member on training data:")
            
            # compute LogLoss for this ensemble member, on validation data
            cat("\nLogLoss of this ensemble member on validation data:")
            
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
        cat("\nLogLosses of ensemble on training data so far:")
        cat("\nMean LogLoss of ensemble on training data so far:")
        
        cat("\nLogLosses of ensemble on validation data so far:")
        cat("\nMean LogLoss of ensemble on validation data so far:")
        
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
    #submission <- read.csv(path_submission)
    #reshape predictions into 1D
    fs <- t(as.matrix(final_submission))
    dim(fs) <- c(prod(dim(fs)),1)
    submission[,2] <- fs #replace 0s with actual predictions
    #write.csv(submission, file = "./submission.csv", quote = F, row.names = F)
    sink()
}
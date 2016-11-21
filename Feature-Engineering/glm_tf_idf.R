#Home Depot Product Search Relevance

#Make predictions on tf_idf using regularised linear regression. Save out-of-fold training predictions and test predictions to be used
#as a feature in various stage 0 models

#Authors: Tyrone Cragg & Liam Culligan

#Date: April 2016

#Load required packages
library(Matrix)
library(caret)
library(caTools)
library(dplyr)
library(data.table)

#Read in required functions
source("functions.R")

#Load required data
load("feature_engineering_tf_idf.RData")

#Extract and remove id
trainID = train_tf_idf[, id]
testID = test_tf_idf[, id]

train_tf_idf[, id := NULL]
test_tf_idf[, id := NULL]

#Create Sparse Matrices 
trainM = sparse.model.matrix(~. -1, data = train_tf_idf)
testM = sparse.model.matrix(~. -1, data = test_tf_idf)

#Create Validation Set -- train on 1/3 and validate on 2/3 to mimic train:test split

#Assign seeds for repeats
seeds = c(44, 55, 66, 77, 88)

#Initialise the data frame Results - will be used to save the average results for each set of parameters of a grid search
Results = data.frame(alpha = NA, lambda = NA, RMSE = NA)

#Initialise the data frame TuningResults - will be used to save the results for each repeat within the grid search
TuningResults = data.frame(alpha = NA, lambda = NA, rep = NA, RMSE = NA)

#Initialise the data frame OOSPreds - will be used to save the out-of-fold model predictions
OOSPreds = data.frame(pred = NA, obs = NA, rowIndex = NA, alpha = NA, lambda = NA, Resample = NA)

#Set the parameters to be used within the grid search
alpha_values = 1 #Apply the lasso penalty
lambda_values = c(0.00001, 0.0001, 0.001, 0.01)

#Loop through all combinations of the parameters to be used in the above grid
for (alpha in alpha_values) {
  for (lambda in lambda_values) {
    
    #Set up tuning grid
    tuneGrid = expand.grid(alpha=alpha, lambda=lambda)
    
    #Repeat loop 5 times with 5 diffrent seeds
    for (rep in 1:5) {
      
      #Get a vector of row numbers
      row = 1:nrow(trainM)
      
      #Set seed for this repeat from the vector seeds
      seed = seeds[rep]
      
      #Split sample based on target variable
      set.seed(seed)
      fold_rows_1_2 = sample.split(relevance_tf_idf, SplitRatio=2/3)
      
      #Row numbers for which sample.split returned FALSE (i.e. 1/3 of training data)
      train_fold_3  = row[fold_rows_1_2 == F]
      
      #Row numbers for which sample.split returned TRUE (i.e. 2/3 of training data)
      train_folds_1_2 = row[fold_rows_1_2 == T]
      
      #Split remaining 2/3 of training data in half to create remaining 2 folds of 1/3 of training data each
      set.seed(seed)
      fold_rows_1 = sample.split(relevance_tf_idf[train_folds_1_2], SplitRatio=0.5)
      
      #Row numbers for which sample.split returned TRUE (i.e. 1/3 of training data)
      train_fold_1  = train_folds_1_2[fold_rows_1 == T]
      
      #Row numbers for which sample.split returned FALSE (i.e. 1/3 of training data)
      train_fold_2  = train_folds_1_2[fold_rows_1 == F]
      
      #Create index of row numbers to be trained on for each fold (1/3 of training data)
      index = list(c(train_fold_1), c(train_fold_2), c(train_fold_3))
      
      #Create index of row numbers to be validated on for each fold (2/3 of training data)
      indexOut = list(c(train_fold_2, train_fold_3), c(train_fold_1, train_fold_3), c(train_fold_1, train_fold_2))
      
      #Create trainControl object for Caret using created fold indices
      trainControl = trainControl(method="LGOCV",
                                  summaryFunction=defaultSummary,
                                  predictionBounds=c(1,3),
                                  index=index,
                                  indexOut=indexOut,
                                  verbose=T,
                                  savePredictions=T)
      
      #Train model using the custom cross validitian scheme established above
      set.seed(seed)
      Mod = train(x=trainM, y=relevance_tf_idf, method="glmnet", metric="RMSE", trControl=trainControl, tuneGrid=tuneGrid)
      
      #Extract out-of-sample predictions
      PredTrainOOS = Mod$pred
      
      #Add out-of-sample predictions to OOSPreds
      OOSPreds = rbind(OOSPreds, PredTrainOOS)
      
      #Extract RMSE score
      RMSE = Mod$results$RMSE
      
      #Save tuning results for this repeat
      TuningResults = rbind(TuningResults, c(alpha, lambda, rep, RMSE))
      
      rm(Mod)
      gc(reset=T)
      
    }
    
    #Filter tuning results for these tuning parameters for all repeats
    Tuning = subset(TuningResults, TuningResults$alpha == alpha & TuningResults$lambda == lambda)
    
    #Calculate average RMSE for these tuning parameters across all repeats
    TuningRMSE = mean(Tuning$RMSE)
    
    #Add tuning results to Results
    Results = rbind(Results, c(alpha, lambda, TuningRMSE))
    
  }
}

#Order Results by ascending order of RMSE
Results = na.omit(Results)
Results = Results[order(Results$RMSE),]

#Filter OOSPreds for the best tuning parameters
OOSPreds = na.omit(OOSPreds)
OOSPreds = subset(OOSPreds, alpha == Results$alpha[1] & lambda == Results$lambda[1])

#Get out-of-sample predictions for each row averaged over the 5 repeats
train_glm_tf_idf = OOSPreds %>%
  group_by(rowIndex) %>%
  summarise(glm_pred = mean(pred)) %>%
  as.data.table()

#Order out-of-sample predictions by row number
train_glm_tf_idf = train_glm_tf_idf[order(train_glm_tf_idf$rowIndex),]
#Add the id to this data.table
train_glm_tf_idf[, id := trainID]
#Remove rowIndex
train_glm_tf_idf[, rowIndex := NULL]

#Set model parameters to the best parameters found from the above grid search
tuneGrid = expand.grid(alpha = Results$alpha[1], lambda = Results$lambda[1])

#Train a model using the entire training set
trainControl = trainControl(method="none",
                            summaryFunction=defaultSummary,
                            predictionBounds=c(1,3),
                            verbose=T)

set.seed(seed)
ModUpdate = train(x = trainM, y = relevance_tf_idf, method = "glmnet", metric = "RMSE", trControl = trainControl, tuneGrid = tuneGrid)

#Make predictions on test
predTest = predict(ModUpdate, testM, type = "raw")

test_glm_tf_idf = data.table(id = testID, glm_pred = predTest)

#Save out-of-sample training and testing predictions as RData
save(list = c("train_glm_tf_idf", "test_glm_tf_idf"), file = "glm_tf_idf.RData")

#Detach all packages from session to avoid conficts with other scripts
detachAllPackages()

#Remove all objects from the workspace
rm(list = ls())
gc()

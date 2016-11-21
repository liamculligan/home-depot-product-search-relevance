#Home Depot Product Search Relevance

#Train multiple regularised linear models on feature_engineering_set_1_reduced using a custom cross-validation scheme and bag the results

#Authors: Tyrone Cragg & Liam Culligan

#Date: April 2016

#Load required packages
library(data.table)
library(dplyr)
library(caret)
library(caTools)

#Read in required functions
source("functions.R")

#Load required data
load("feature_engineering_set_1_reduced.RData")

#Rename for convenience
train = copy(train_reduced)
test = copy(test_reduced)
rm(train_reduced, test_reduced)

#Extract relevance and id
relevance = train[, relevance]
trainID = train[, id]
testID = test[, id]

#Remove relevance and id
train[, ':=' (relevance = NULL,
              id = NULL)]
test[, id := NULL]

#Replace NAs with -1
train[is.na(train)] = -1
test[is.na(test)] = -1

#Check the difference between distributions of search_term_overlap and search_term_relevance between train and test.
#Approximately 89% of search terms in train are found in test but only 29% of search terms in test are found in train.
#We want the same distribution of search_term_overlap and search_term_relevance in train as in test so that the model generalises well.
#We will therefore randomly remove some search_term_overlap and search_term_relevance values from train. This process is repeated 10 times
#with different observations being randomly removed each time.
mean(train$search_term_overlap)
mean(test$search_term_overlap)

#Convert test to a matrix (train will be transformed later)
testM = model.matrix(~. -1, data = test)

#Create Validation Set -- train on 1/3 and validate on 2/3 to mimic train:test split

#Assign seeds for repeats
seeds = c(44, 55, 66, 77, 88, 99, 111, 222, 333, 444)

#Initialise the data frame Results - will be used to save the average results for each set of parameters of a grid search
Results = data.frame(alpha = NA, lambda = NA, RMSE = NA)

#Initialise the data frame TuningResults - will be used to save the results for each repeat within the grid search
TuningResults = data.frame(alpha = NA, lambda = NA, rep = NA, RMSE = NA)

#Initialise the data frame OOSPreds - will be used to save the out-of-fold model predictions
OOSPreds = data.frame(pred = NA, obs = NA, rowIndex = NA, alpha = NA, lambda = NA, Resample = NA)

#Set the parameters to be used within the grid search
alpha_values = c(0, 0.5, 1)
lambda_values = c(0.00001, 0.001, 0.1)

#Loop through all combinations of the parameters to be used in the above grid
for (alpha_value in alpha_values) {
  for (lambda_value in lambda_values) {
    
    #Set up tuning grid
    tuneGrid = expand.grid(alpha = alpha_value, lambda = lambda_value)
    
    #Repeat loop 10 times with 10 diffrent seeds
    for (rep in 1:10) {
      
      #Get a vector of row numbers
      row = 1:nrow(train)
      
      #Set seed for this repeat from the vector seeds
      seed = seeds[rep]
      
      #Create a copy of train for this repeat
      trainRep = copy(train)
      
      #Stratify by search_term_overlap
      set.seed(seed)
      split = as.numeric(sample.split(trainRep$search_term_overlap, SplitRatio=0.33))
      
      #Randomly remove 67% of search_term_overlap and search_term_relevance values from train so that these variable have the same distribution
      #in train and test. Without this the model will overfit.
      trainRep$search_term_overlap = trainRep$search_term_overlap*split
      trainRep$search_term_relevance[trainRep$search_term_overlap == 0] = -1
      
      #Convert trainRep to a matrix
      trainM = model.matrix(~. -1, data = trainRep)
      
      #Split sample based on target variable
      set.seed(seed)
      fold_rows_1_2 = sample.split(relevance, SplitRatio=2/3)
      
      #Row numbers for which sample.split returned FALSE (i.e. 1/3 of training data)
      train_fold_3  = row[fold_rows_1_2==F]
      
      #Row numbers for which sample.split returned TRUE (i.e. 2/3 of training data)
      train_folds_1_2 = row[fold_rows_1_2==T]
      
      #Split remaining 2/3 of training data in half to create remaining 2 folds of 1/3 of training data each
      set.seed(seed)
      fold_rows_1 = sample.split(relevance[train_folds_1_2], SplitRatio=0.5)
      
      #Row numbers for which sample.split returned TRUE (i.e. 1/3 of training data)
      train_fold_1  = train_folds_1_2[fold_rows_1==T]
      
      #Row numbers for which sample.split returned FALSE (i.e. 1/3 of training data)
      train_fold_2  = train_folds_1_2[fold_rows_1==F]
      
      #Create index of row numbers to be trained on for each fold (1/3 of training data)
      index = list(c(train_fold_1), c(train_fold_2), c(train_fold_3))
      
      #Create index of row numbers to be validated on for each fold (2/3 of training data)
      indexOut = list(c(train_fold_2, train_fold_3), c(train_fold_1, train_fold_3), c(train_fold_1, train_fold_2))
      
      #Create trainControl object for Caret using created fold indices
      trainControl = trainControl(method="LGOCV",
                                  summaryFunction=defaultSummary,
                                  index=index,
                                  indexOut=indexOut,
                                  verbose=T,
                                  savePredictions=T)
      
      #Train model using the custom cross validitian scheme established above
      set.seed(seed)
      Mod = train(x = trainM, y = relevance, method = "glmnet", trControl = trainControl, tuneGrid = tuneGrid, metric = "RMSE")
      
      #Extract out-of-sample predictions
      PredTrainOOS = Mod$pred
      
      #Add out-of-sample predictions to OOSPreds
      OOSPreds = rbind(OOSPreds, PredTrainOOS)
      
      #Extract RMSE score
      RMSE = Mod$results$RMSE
      
      #Save tuning results for this repeat
      TuningResults = rbind(TuningResults, c(alpha_value, lambda_value, rep, RMSE))
      TuningResults = na.omit(TuningResults)
      
      rm(Mod)
      gc(reset=T)
      
    }
    
    #Filter tuning results for these tuning parameters for all repeats
    Tuning = subset(TuningResults, TuningResults$alpha == alpha_value & TuningResults$lambda == lambda_value)
    
    #Calculate average RMSE for these tuning parameters across all repeats
    TuningRMSE = mean(Tuning$RMSE)
    
    #Add tuning results to Results
    Results = rbind(Results, c(alpha_value, lambda_value, TuningRMSE))
    
  }
}

#Order Results by ascending order of RMSE
Results = na.omit(Results)
Results = Results[order(Results$RMSE),]

#Filter OOSPreds for the best tuning parameters
OOSPreds = na.omit(OOSPreds)
OOSPreds = subset(OOSPreds, alpha == Results$alpha[1] & lambda == Results$lambda[1])

#Get out-of-sample predictions for each row averaged over the 10 repeats
train_glm = OOSPreds %>%
  group_by(rowIndex) %>%
  summarise(relevance = mean(pred)) %>%
  arrange(rowIndex) %>%
  as.data.table()

#Add the id to this data.table
train_glm[, id := trainID]
#Remove rowIndex
train_glm[, rowIndex := NULL]

#Save the version number for the stacked generalisation
Version = 2

#Save the out-of-fold predictions to be used in the stacked generalisation
Filename = paste0("Train-V", Version, ".csv")
PredColname = paste0("V", Version)

#Set column names
colnames(train_glm) = c(PredColname, "id")

#Save CSV
write.csv(train_glm, Filename, row.names = F)

#Train the model on the full training set using the best parameters
tuneGrid = expand.grid(alpha = Results$alpha[1], lambda = Results$lambda[1])

trainControl = trainControl(method = "none",
                            summaryFunction = defaultSummary,
                            verbose = T)

#Initiliase an empty data frame to save the predictions for each repeat of the model
PredTestBag = data.frame(row = 1:nrow(test))

#Repeat loop 10 times with 10 diffrent seeds
for (rep in 1:10) {     
  
  #Set seed for this repeat from the vector seeds
  seed = seeds[rep]
  
  #Create a copy of train for this repeat
  trainRep = copy(train)
  
  #Stratify by search_term_overlap
  set.seed(seed)
  split = as.numeric(sample.split(trainRep$search_term_overlap, SplitRatio=0.33))
  
  #Randomly remove 67% of search_term_overlap and search_term_relevance values from train so that these variable have the same distribution
  #in train and test. Without this the model will overfit.
  trainRep$search_term_overlap = trainRep$search_term_overlap*split
  trainRep$search_term_relevance[trainRep$search_term_overlap == 0] = -1
  
  #Convert trainRep to a matrix
  trainM = model.matrix(~. -1, data=trainRep)
  
  #Train the model using the full training set
  set.seed(seed)
  Mod = train(x = trainM, y = relevance, method = "glmnet", trControl = trainControl, tuneGrid = tuneGrid, metric = "RMSE")
  
  #Obtain test set predictions
  PredTest = predict(Mod, testM, type = "raw")
  
  #Save the test set predictions, for this seed, to a data frame
  PredTestBag = cbind(PredTestBag, PredTest)
  
}

#Remove row number
PredTestBag$row = NULL

#Get the average predictions for the different models
PredTest = rowMeans(PredTestBag)

#Save the test predictions to be used in the stacked generalisation
test_glm = data.frame(relevance = PredTest, id = testID)

Filename = paste0("Test-V", Version, ".csv")
PredColname = paste0("V", Version)

#Set column names
colnames(test_glm) = c(PredColname, "id")

#Save CSV
write.csv(test_glm, Filename, row.names = F)

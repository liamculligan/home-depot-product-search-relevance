#Home Depot Product Search Relevance

#Train a stage 1 generaliser (gradient boosted decision tree) using stage 0 out-of-fold model predictions as inputs

#Authors: Tyrone Cragg & Liam Culligan

#Date: April 2016

#Load required packages
library(data.table)
library(corrplot)
library(xgboost)

#Load ids and target variables
train = fread("train.csv", select = c("id", "relevance"))
test = fread("test.csv", select = "id")

#Set required keys prior to joining
setkey(train, id)
setkey(test, id)

#Read in stage 0 model predictions to create stage 1 data

for (version in 1:7) {
  
  trainFilename = paste("Train-V", version, ".csv", sep="")
  testFilename = paste("Test-V", version, ".csv", sep="")
  
  trainImport = fread(trainFilename)
  testImport = fread(testFilename)
  
  setkey(trainImport, id)
  setkey(testImport, id)
  
  train = train[trainImport]
  test = test[testImport]
  
}

#Extract relevance and id
relevance = train[, relevance]
trainID = train[, id]
testID = test[, id]

#Remove relevance and id
train[, ':=' (relevance = NULL,
              id = NULL)]
test[, id := NULL]

#Create a correlation plot of stage 1 data - stacked generalisation performs better when models are not highly correlated
M = cor(train)
M[upper.tri(M)] = 0
diag(M) = 0
corrplot(M, method = "square")

#Extract column names
feature_names = names(train)

#Create Validation Set -- train on 1/3 and validate on 2/3 to mimic train:test split

#Assign seeds for repeats
seeds = c(44, 55, 66, 77, 88, 99, 111, 222, 333, 444)

#Initialise the data frame Results - will be used to save the average results for each set of parameters of a grid search
Results = data.frame(eta = NA, max_depth = NA, subsample = NA, colsample_bytree = NA, RMSE = NA, n_rounds = NA)

#Initialise the data frame TuningResults - will be used to save the results for each repeat within the grid search
TuningResults = data.frame(eta = NA, max_depth = NA, subsample = NA, colsample_bytree = NA, rep = NA, loop = NA, RMSE = NA, n_rounds = NA)

#Initialise the data frame OOSPreds - will be used to save the out-of-fold model predictions
OOSPreds = data.frame(id = NA, relevance = NA, eta = NA, max_depth = NA, subsample = NA, colsample_bytree = NA, rep = NA)

#Fix the early stop round for boosting
early_stop_round = 10

#Set the parameters to be used within the grid search
eta_values = c(0.02)
max_depth_values = c(3) 
subsample_values = c(0.4) 
colsample_bytree_values = c(0.6) 

#Loop through all combinations of the parameters to be used in the above grid
for (eta_value in eta_values) {
  for (max_depth_value in max_depth_values) {
    for (subsample_value in subsample_values) {
      for (colsample_bytree_value in colsample_bytree_values) {
        
        #Set the parameters to be used within the grid search - only best set of parameters found via tuning is retained here
        param = list(objective = "reg:linear",
                     booster = "gbtree",
                     eval_metric = "rmse",
                     eta = eta_value,
                     max_depth = max_depth_value,
                     subsample = subsample_value,
                     colsample_bytree = colsample_bytree_value)
        
        #Repeat loop 10 times with 10 diffrent seeds
        for (rep in 1:10) {
          
          #Set seed for this repeat from the vector seeds
          seed = seeds[rep]
          
          #Create a copy of train for this repeat
          trainRep = copy(train)
          
          #Split sample based on target variable
          set.seed(seed)
          fold_rows_1_2 = sample.split(relevance, SplitRatio=2/3)
          
          #Observations for which sample.split returned FALSE (i.e. 1/3 of training data)
          train_fold_3  = trainRep[fold_rows_1_2 == F, ]
          relevance_fold_3 = relevance[fold_rows_1_2 == F]
          id_fold_3 = trainID[fold_rows_1_2 == F]
          
          #Observations for which sample.split returned TRUE (i.e. 2/3 of training data)
          train_folds_1_2 = trainRep[fold_rows_1_2 == T, ]
          relevance_folds_1_2 = relevance[fold_rows_1_2 == T]
          id_folds_1_2 = trainID[fold_rows_1_2 == T]
          
          #Split remaining 2/3 of training data in half to create remaining 2 folds of 1/3 of training data each
          set.seed(seed)
          fold_rows_1 = sample.split(relevance_folds_1_2, SplitRatio=0.5)
          
          #Observations for which sample.split returned TRUE (i.e. 1/3 of training data)
          train_fold_1  = train_folds_1_2[fold_rows_1 == T, ]
          relevance_fold_1 = relevance_folds_1_2[fold_rows_1==T]
          id_fold_1 = id_folds_1_2[fold_rows_1==T]
          
          #Observations for which sample.split returned FALSE (i.e. 1/3 of training data)
          train_fold_2  = train_folds_1_2[fold_rows_1 == F, ]
          relevance_fold_2 = relevance_folds_1_2[fold_rows_1==F]
          id_fold_2 = id_folds_1_2[fold_rows_1==F]
          
          #Loop 3 times to train on 1 fold and validate on other 2 folds, using each fold as a training set once
          for (loop in 1:3) {
            
            #Set up training and validation matrices for each loop
            if (loop == 1) {
              
              #Convert the training fold feature set and target variable to an xgb.DMatrix
              dtrain = xgb.DMatrix(data = data.matrix(train_fold_1),label = relevance_fold_1, missing = -999999)
              
              #Convert the validation folds feature sets and target variable to an xgb.DMatrix
              dval = xgb.DMatrix(data = data.matrix(rbind(train_fold_2, train_fold_3)),
                                 label = c(relevance_fold_2, relevance_fold_3), missing = -999999)
              
              #Return the validation fold ids as a vector
              valID = c(id_fold_2, id_fold_3)
              
            } else if (loop == 2) {
              
              dtrain = xgb.DMatrix(data = data.matrix(train_fold_2),label = relevance_fold_2, missing = -999999)
              
              dval = xgb.DMatrix(data = data.matrix(rbind(train_fold_1, train_fold_3)),
                                 label = c(relevance_fold_1, relevance_fold_3), missing = -999999)
              
              valID = c(id_fold_1, id_fold_3)
              
            } else if (loop == 3) {
              
              dtrain = xgb.DMatrix(data = data.matrix(train_fold_3),label = relevance_fold_3, missing = -999999)
              
              dval = xgb.DMatrix(data = data.matrix(rbind(train_fold_1, train_fold_2)),
                                 label = c(relevance_fold_1, relevance_fold_2), missing = -999999)
              
              valID = c(id_fold_1, id_fold_2)
              
            }
            
            #Now explicity specify the training and validation sets for xgboost evaluation
            watchlist = list(val = dval, train = dtrain)
            
            #Train model
            set.seed(seed)
            XGBtrain = xgb.train(params = param,
                                 data = dtrain,
                                 nrounds = 10000,
                                 verbose = 1,
                                 early.stop.round = early_stop_round,
                                 watchlist = watchlist,
                                 maximize = F
            )
            
            #Extract RMSE score
            RMSE = XGBtrain$bestScore
            
            #Extract the number of boosting rounds for this set of parameters
            n_rounds = XGBtrain$bestInd
            
            #Save tuning results for this repeat
            TuningResults = rbind(TuningResults, c(eta_value, max_depth_value, subsample_value,
                                                   colsample_bytree_value, rep, loop, RMSE, n_rounds))
            
            rm(XGBtrain)
            gc(reset=T)
            
          }
        }
        
        #Filter tuning results for these tuning parameters for all repeats
        Tuning = subset(TuningResults, TuningResults$eta == eta_value & TuningResults$max_depth == max_depth_value &
                          TuningResults$subsample == subsample_value &
                          TuningResults$colsample_bytree == colsample_bytree_value)
        
        #Calculate average RMSE and n_rounds for these tuning parameters across all repeats
        TuningRMSE = mean(Tuning$RMSE)
        TuningNRounds = mean(Tuning$n_rounds)
        
        #Add tuning results to Results
        Results = rbind(Results, c(eta_value, max_depth_value, subsample_value, colsample_bytree_value, TuningRMSE, TuningNRounds))
        
      }
    }
  }
}

#Order Results by ascending order of RMSE
Results = na.omit(Results)
Results = Results[order(Results$RMSE),]

#The number of boosting rounds is a function of training size. The number of rounds for the above cross-validation scheme is therefore
#insufficient. Rerun cross-validation using the best parameters found above, but use repeated" 5-fold cross-validation to only select
#the number of boosting rounds for training on the full data set.
param = list(objective = "reg:linear",
             booster = "gbtree",
             eval_metric = "rmse",
             eta = Results$eta[1],
             max_depth = Results$max_depth[1],
             subsample = Results$subsample[1],
             colsample_bytree = Results$colsample_bytree[1]
)

#Initialise n_rounds_sum
n_rounds_sum = 0

#Repeat 3 times
for (rep in 1:3) {     
  
  #Set seed for this repeat from the vector seeds
  seed = seeds[rep]
  
  #Randomly remove some search term relevance values
  trainRep = copy(train)
  
  #Convert the feature set and target variable to an xgb.DMatrix
  dtrain = xgb.DMatrix(data = data.matrix(trainRep), label = relevance, missing = -999999)
  
  #Train the model
  set.seed(seed)
  XGBcv = xgb.cv(params = param, 
                 data = dtrain,
                 nrounds = 10000,
                 verbose = T,
                 nfold = 5,
                 early.stop.round = early_stop_round
  )
  
  #Add n_rounds to n_rounds_sum
  n_rounds_sum = n_rounds_sum + length(XGBcv$test.rmse.mean) - early_stop_round
  
}

#Calculate average n_rounds
n_rounds = round(n_rounds_sum/3)

#Initiliase an empty data frame to save the predictions for each repeat of the model
PredTestBag = data.frame(row = 1:nrow(test))

#Repeat loop 10 times with 10 diffrent seeds and bag the model predictions
for (rep in 1:10) {     
  
  #Set seed for this repeat from the vector seeds
  seed = seeds[rep]
  
  #Create a copy of train for this repeat
  trainRep = copy(train)
  
  #Convert the feature set and target variable to an xgb.DMatrix
  dtrain = xgb.DMatrix(data=data.matrix(trainRep), label=relevance, missing = -999999)
  
  #Train the model using the full training set
  set.seed(seed)
  XGB = xgboost(params = param, 
                data = dtrain,
                nrounds = n_rounds,
                verbose = 1
  )
  
  #Obtain test set predictions
  PredTest = predict(XGB, data.matrix(test[, feature_names, with = F]), missing = -999999)
  
  #Save the test set predictions, for this seed, to a data frame
  PredTestBag = cbind(PredTestBag, PredTest)
  
}

#Remove row number
PredTestBag$row = NULL

#Get the average predictions for the different models
PredTest = rowMeans(PredTestBag)

#Create Submission File and Write CSV
Submission = data.frame(id = testID, relevance = PredTest)
write.csv(Submission, "XGB_Stage_1_test.csv", row.names = F)

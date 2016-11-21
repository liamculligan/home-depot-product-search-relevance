# Home Depot Product Search Relevance
[Kaggle - Predict the relevance of search results] (https://www.kaggle.com/c/home-depot-product-search-relevance)

## Introduction
The goal of this competition was to predict the relevance between pairs of real customer searches on [Home Depot's website](http://www.homedepot.com) and the product returned for each search. <br> The feature set provided consisted of searches, the names of the products returned for each search, a text description of the products and technical specifications of the proudcts, where available. <br>
Each query had an associated relevance score between 1 (not relevant) and 3 (highly relevant), which were obtained by human evaluation. <br> Model predictions were evaluated using [root mean squared error](https://www.kaggle.com/wiki/RootMeanSquaredError).

## Team Meambers
The team, Arrested Development, consisted of [Tyrone Cragg] (https://github.com/tyronecragg) and [Liam Culligan] (https://github.com/liamculligan).

## Solution Architecture
![Solution Architecture](https://github.com/liamculligan/home-depot-product-search-relevance/blob/master/Solution-Architecture.jpg?raw=true "Solution Architecture")

## Performance
The solution obtained a rank of [74th out of 2125 teams] (https://www.kaggle.com/c/bosch-production-line-performance/leaderboard/private) with a private leaderboard root mean squared error of 0.45673. <br> The cross-validated root mean squared error for the final model was 0.44533.

## Execution
1. Create a working directory for the project <br>
2. [Download the data from Kaggle] (https://www.kaggle.com/c/home-depot-product-search-relevance/data) and place in the working directory
3. Run pre-processing scripts: <br>
3.1 `spelling_corrector.py` <br>
3.2 `pre_process.R` <br>
4. Run feature engineering scripts: <br>
4.1 `feature_engineering_tf_idf.R` <br>
4.2 `glm_tf_idf.R` <br>
4.3 `feature_engineering_ngrams.R` <br>
4.4 `feature_engineering_main.R` <br>
5. Run the Stage 0 model scripts for the stacked generalisation: <br>
5.1 `KNN1.R`<br>
5.2 `GLM1.R` <br>
5.3 `XGB1.R` <br>
5.4 `XGB2.R` <br>
5.5 `XGB3.R` <br>
5.6 `XGB4.R` <br>
5.7 `XGB5.R` <br>
6. Run the Stage 1 model script, `XGB.R`

## Requirements
* R 3+
* Python 3+

# TO DO
Finish

#Home Depot Product Search Relevance

#Feature engineering - create various features relating to ngram matches between search_term, product_title and product_description

#Original author: Maher Harb
#Updated by: Tyrone Cragg & Liam Culligan

#Date: April 2016

#Load required packages
library(data.table)
library(RWeka)
library(stringdist)
library(combinat)

#Read in required functions
source("functions.R")

load("pre_process.RData")

#Select only the columns required
data = data[, .(id, search_term, product_title, product_description)]

#Initialise data.table
ngram_data = data.table(search_term_length = NA, product_title_length = NA, product_description_length = NA, product_title_1grams = NA,
                        product_title_2grams = NA, product_title_3grams = NA, product_title_4grams = NA, product_description_1grams = NA,
                        product_description_2grams = NA, product_description_3grams = NA, product_description_4grams = NA,
                        product_title_ngram_matchscore = NA, product_description_ngram_matchscore = NA, product_title_ngram_simscore = NA,
                        product_description_ngram_simscore = NA, product_title_2grams_2ndorder = NA, product_description_2grams_2ndorder = NA,
                        product_title_similarity = NA, product_description_similarity = NA)
ngram_data = na.omit(ngram_data)

#Loop through data and apply ngramMatches function to each row
for (i in 1:nrow(data)) {
  print(i)
  
  temp = data.table(t(ngramMatches(data[i, search_term], data[i, product_title], data[i, product_description])))
  
  #Add row to ngram_data
  ngram_data = rbindlist(list(ngram_data, temp), use.names=F)
  
}

#If there is only a unigram for a search term it cannot be shuffled to create new bigrams.

#Impute with mean to fill these missing values
ngram_data[is.na(ngram_data[, product_title_2grams_2ndorder]),
           product_title_2grams_2ndorder := mean(ngram_data[, product_title_2grams_2ndorder], na.rm=T)]

#Impute empty description
#Find rows without a product_description (length = 0)
rows = (ngram_data[, product_description_length] == 0)

#Set these rows to NA for all product_description features
ngram_data[rows, product_description_2grams_2ndorder := NA]
ngram_data[is.na(ngram_data[, product_description_2grams_2ndorder]),
           product_description_2grams_2ndorder := mean(ngram_data[, product_description_2grams_2ndorder], na.rm=T)]
ngram_data[rows, product_description_1grams := NA]
ngram_data[rows, product_description_2grams := NA]
ngram_data[rows, product_description_3grams := NA]
ngram_data[rows, product_description_4grams := NA]
ngram_data[rows, product_description_ngram_matchscore := NA]
ngram_data[rows, product_description_ngram_simscore := NA]
ngram_data[rows, product_description_similarity := NA]

#Impute these rows with mean values
ngram_data[rows, product_description_1grams := mean(ngram_data[, product_description_1grams], na.rm=T)]
ngram_data[rows, product_description_2grams := mean(ngram_data[, product_description_2grams], na.rm=T)]
ngram_data[rows, product_description_3grams := mean(ngram_data[, product_description_3grams], na.rm=T)]
ngram_data[rows, product_description_4grams := mean(ngram_data[, product_description_4grams], na.rm=T)]
ngram_data[rows, product_description_ngram_matchscore := mean(ngram_data[, product_description_ngram_matchscore], na.rm=T)]
ngram_data[rows, product_description_ngram_simscore := mean(ngram_data[, product_description_ngram_simscore], na.rm=T)]
ngram_data[rows, product_description_similarity := mean(ngram_data[, product_description_similarity], na.rm=T)]



#Normalise features by number of words in search term to enhance correlation between features and target variable

#Normalise by number of words in search term
ngram_data[, product_title_1grams := ngram_data[, product_title_1grams]/ngram_data[, search_term_length]]
ngram_data[, product_title_2grams := ngram_data[, product_title_2grams]/ngram_data[, search_term_length]]
ngram_data[, product_title_3grams := ngram_data[, product_title_3grams]/ngram_data[, search_term_length]]
ngram_data[, product_title_4grams := ngram_data[, product_title_4grams]/ngram_data[, search_term_length]]
ngram_data[, product_description_1grams := ngram_data[, product_description_1grams]/ngram_data[, search_term_length]]
ngram_data[, product_description_2grams := ngram_data[, product_description_2grams]/ngram_data[, search_term_length]]
ngram_data[, product_description_3grams := ngram_data[, product_description_3grams]/ngram_data[, search_term_length]]
ngram_data[, product_description_4grams := ngram_data[, product_description_4grams]/ngram_data[, search_term_length]]

#Improves correlation with target variable
ngram_data[, product_description_ngram_matchscore := ngram_data[, product_description_ngram_matchscore]^2]
ngram_data[, product_title_2grams_2ndorder := ngram_data[, product_title_2grams_2ndorder]^0.5]
ngram_data[, length_ratio := ngram_data[, search_term_length]/ngram_data[, product_title_length]]

#Remove unnecessary features
ngram_data[, ':=' (product_title_similarity = NULL,
                   product_description_ngram_simscore = NULL,
                   product_description_similarity = NULL,
                   search_term_length = NULL,
                   product_title_length = NULL,
                   product_description_length = NULL)]

#Add id to ngram_data
ngram_data[, id := data[, id]]

#Save ngram_data
save(ngram_data, file="ngram_data.RData")

#Detach all packages from session to avoid conficts with other scripts
detachAllPackages()

#Remove all objects from the workspace
rm(list = ls())
gc()

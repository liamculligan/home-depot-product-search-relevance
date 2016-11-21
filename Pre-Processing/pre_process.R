#Home Depot Product Search Relevance

#Pre-process data

#Authors: Tyrone Cragg & Liam Culligan

#Date: April 2016

#Load required packages
library(data.table)
library(dplyr)
library(dtplyr)
library(tm)

#Read in required functions
source("functions.R") 

#Read in data 
train = fread("train.csv", encoding = "Latin-1")
test = fread("test.csv", encoding = "Latin-1")
productDescriptions = fread("product_descriptions.csv", encoding = "Latin-1")
attributes = fread("attributes.csv", encoding = "Latin-1")

#Add placeholder target variable to test
test[, relevance := NA]

#Row bind train and test
data = rbind(train, test)

#Set appropriate keys prior to joining
setkey(data, "product_uid")
setkey(productDescriptions, "product_uid")

#Join data with product descriptions
data = productDescriptions[data]

#Transform attributes to a tidy format - one row for each product_uid
attributesNameValue = attributes %>% 
  group_by(product_uid) %>%
  summarise(name = paste(name, collapse=". "), value = paste(value, collapse=". "))

attributesNameValue[, ':=' (name = paste0(". ", name, "."),
                            value = paste0(". ", value, "."))]

#Set appropriate key prior to joining
setkey(attributesNameValue, "product_uid")

#Join data with attributes
data = attributesNameValue[data]

#Clean product_description and title
#If words are concatenated incorrectly, e.g. aA, separate to a A

cols = c("product_description", "product_title")
data[, (cols) := lapply(.SD, split_text), .SDcols = cols]

#Correct speling errors and typos in search - from "spelling_corrector.py"
corrections = fread("corrections.csv", encoding = "Latin-1")

#Correct spelling errors and typos in search_term
data[, search_term := correct_spelling(search_term, corrections$search_term_old, corrections$search_term_new)]

#Convert all strings to lowercase
character_cols = sapply(data, class) == "character"
character_names = names(data)[character_cols]

data[, (character_names) := lapply(.SD, tolower), .SDcols = character_names]

#Remove punctuation and whitespaces from character variables
data[, (character_names) := lapply(.SD, remove_punctuation_whitespace), .SDcols = character_names]

#Standardise units of measurement
cols = c("product_description", "product_title", "search_term", "value")

#A vector of all units of measurement to be standardised
units = c("foot|feet|ft", "cu foot|cu feet|cu ft|cubic foot|cubic feet|cubic ft", "inches|inch|in", 
          "pounds|pound|lbs|lb", "square foot|square feet|square ft|sq foot|sq feet|sq ft",
          "gallons|gallon|gal", "ounces|ounce|oz", "centimeter|centimeters|centimetre|centimetres|cm",
          "millimeter|millimeters|millimetre|millimetres|mm", "degrees|degree",
          "volts|volt|v", "hertz|hert|hz", "watts|watt", "amperes|ampere|amps|amp")

#A vector with the all standardised units of measurements
units_standard = c("ft", "cuft", "inch", "lb", "sqft", "gal", "oz", "cm", "mm", "degree", "volt", "hz", "watt", "amp")

#For each character column, standardise the units of measurement using the above vectors
for (i in 1:length(units)) {
  print(i)
  unit = units[i]
  unit_standard = units_standard[i]
  
  data[, (cols) := lapply(.SD, standardise_units, unit, unit_standard), .SDcols = cols]
  
}

#Standardise measurements
data[, (cols) := lapply(.SD, standardise_measurements), .SDcols = cols]

#Save train and test as RData
save(list = c("data", "attributes"), file = "pre_process.RData")

#Detach all packages from session to avoid conficts with other scripts
detachAllPackages()

#Remove all objects from the workspace
rm(list = ls())
gc()

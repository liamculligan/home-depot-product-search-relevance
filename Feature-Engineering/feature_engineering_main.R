#Home Depot Product Search Relevance

#Feature engineering

#Authors: Tyrone Cragg & Liam Culligan

#Date: April 2016

#Load required packages
library(data.table)
library(dplyr)
library(dtplyr)
library(tm)
library(stringr)
library(NLP)
library(openNLP)
library(reshape)
library(stringdist)
library(SnowballC)

#Read in required functions
source("functions.R") 

#Load pre-processed data
load("pre_process.RData")

#Create a variable indicating which products do not have attributes
data[, has_attributes := ifelse(is.na(name), 0, 1)]

#Identifying parts of speech and creating features for the different parts of speech

#Need sentence and word token annotations
sentence_token_annotator = Maxent_Sent_Token_Annotator() #Computes sentence annotations
word_token_annotator = Maxent_Word_Token_Annotator() #Computes word token annotations
pos_tag_annotator = Maxent_POS_Tag_Annotator() #Computes POS tag annotations

variables = c("search_term", "product_title")

for (variable in variables) {
  
  print(variable)
  
  #Create variable names for the different parts of speech considered
  NounsVar = paste(variable, "nouns", sep="_")
  VerbsVar = paste(variable, "verbs", sep="_")
  AdjectivesVar = paste(variable, "adjectives", sep="_")
  AdverbsVar = paste(variable, "adverbs", sep="_")
  ConjunctionsVar = paste(variable, "conjunctions", sep="_")
  NumeralsVar = paste(variable, "numerals", sep="_")
  
  #Create empty lists for each part of speech
  Nouns = lapply(1:nrow(data), function(x) matrix(NA, nrow=1, ncol=1))
  Verbs = lapply(1:nrow(data), function(x) matrix(NA, nrow=1, ncol=1))
  Adjectives = lapply(1:nrow(data), function(x) matrix(NA, nrow=1, ncol=1))
  Adverbs = lapply(1:nrow(data), function(x) matrix(NA, nrow=1, ncol=1))
  Conjunctions = lapply(1:nrow(data), function(x) matrix(NA, nrow=1, ncol=1))
  Numerals = lapply(1:nrow(data), function(x) matrix(NA, nrow=1, ncol=1))
  
  #Input the sentence and word annotations, from these, tag each word as its most likely part of speech
  a3 = lapply(data[[variable]], function(x) annotate(x, pos_tag_annotator, annotate(x, list(sentence_token_annotator, word_token_annotator))))
  
  #Drop sentence annotations now - only interested in individual words
  a3w = lapply(a3, function(x) subset(x, type == "word"))
  
  #Extract parts of speech
  tags = lapply(a3w, function(x) sapply(x$features, `[[`, "POS"))
  
  #Split each word (by whitespace)
  split = strsplit(data[[variable]], " ")
  
  #Associate each word with its tagged part of speech
  for (i in 1:nrow(data)) {
    
    print(i)
    
    split_i = split[[i]]
    
    tags_i = tags[[i]]
    
    #Place each word into its associtated part of speech list - used later to create count and ratio features
    Nouns[[i]] = split_i[which(tags_i=="NN" | tags_i=="NNS")]
    Verbs[[i]] = split_i[which(tags_i=="VB" | tags_i=="VBD" | tags_i=="VBG" | tags_i=="VBN" | tags_i=="VBP" | tags_i=="VBZ")]
    Adjectives[[i]] = split_i[which(tags_i=="JJ" | tags_i=="JJR" | tags_i=="JJS")]
    Adverbs[[i]] = split_i[which(tags_i=="RB" | tags_i=="RBR" | tags_i=="RBS")]
    Conjunctions[[i]] = split_i[which(tags_i=="CC" | tags_i=="IN")]
    Numerals[[i]] = split_i[which(tags_i=="CD")]
    
  }
  
  #Rename tbe resulting lists
  assign(NounsVar, Nouns)
  assign(VerbsVar, Verbs)
  assign(AdjectivesVar, Adjectives)
  assign(AdverbsVar, Adverbs)
  assign(ConjunctionsVar, Conjunctions)
  assign(NumeralsVar, Numerals)
  
  rm(Nouns)
  rm(Verbs)
  rm(Adjectives)
  rm(Adverbs)
  rm(Conjunctions)
  rm(Numerals)
  gc()
  
}

#Stem parts of speech
search_term_nouns = lapply(search_term_nouns, wordStem)
search_term_verbs = lapply(search_term_verbs, wordStem)
search_term_adjectives = lapply(search_term_adjectives, wordStem)
search_term_adverbs = lapply(search_term_adverbs, wordStem)
search_term_conjunctions = lapply(search_term_conjunctions, wordStem)
search_term_numerals = lapply(search_term_numerals, wordStem)
product_title_nouns = lapply(product_title_nouns, wordStem)
product_title_verbs = lapply(product_title_verbs, wordStem)
product_title_adjectives = lapply(product_title_adjectives, wordStem)
product_title_adverbs = lapply(product_title_adverbs, wordStem)
product_title_conjunctions = lapply(product_title_conjunctions, wordStem)
product_title_numerals = lapply(product_title_numerals, wordStem)

#Initialise vectors of appropriate type and length
number_search_term_nouns = vector(mode="numeric", length=nrow(data))
number_search_term_verbs = vector(mode="numeric", length=nrow(data))
number_search_term_adjectives = vector(mode="numeric", length=nrow(data))
number_search_term_adverbs = vector(mode="numeric", length=nrow(data))
number_search_term_conjunctions = vector(mode="numeric", length=nrow(data))
number_search_term_numerals = vector(mode="numeric", length=nrow(data))

number_product_title_nouns = vector(mode="numeric", length=nrow(data))
number_product_title_verbs = vector(mode="numeric", length=nrow(data))
number_product_title_adjectives = vector(mode="numeric", length=nrow(data))
number_product_title_adverbs = vector(mode="numeric", length=nrow(data))
number_product_title_conjunctions = vector(mode="numeric", length=nrow(data))
number_product_title_numerals = vector(mode="numeric", length=nrow(data))

number_match_title_nouns = vector(mode="numeric", length=nrow(data))
number_match_title_verbs = vector(mode="numeric", length=nrow(data))
number_match_title_adjectives = vector(mode="numeric", length=nrow(data))
number_match_title_adverbs = vector(mode="numeric", length=nrow(data))
number_match_title_conjunctions = vector(mode="numeric", length=nrow(data))
number_match_title_numerals = vector(mode="numeric", length=nrow(data))

ratio_match_title_nouns = vector(mode="numeric", length=nrow(data))
ratio_match_title_verbs = vector(mode="numeric", length=nrow(data))
ratio_match_title_adjectives = vector(mode="numeric", length=nrow(data))
ratio_match_title_adverbs = vector(mode="numeric", length=nrow(data))
ratio_match_title_conjunctions = vector(mode="numeric", length=nrow(data))
ratio_match_title_numerals = vector(mode="numeric", length=nrow(data))

#Determine the count by each part of speech and the number of matching words and matching ratio for each part of speech considered

for (i in 1:nrow(data)) {
  print(i)
  
  #Determine the number of words in search term for each part of speech
  number_search_term_nouns[i] = length(search_term_nouns[[i]])
  number_search_term_verbs[i] = length(search_term_verbs[[i]])
  number_search_term_adjectives[i] = length(search_term_adjectives[[i]])
  number_search_term_adverbs[i] = length(search_term_adverbs[[i]])
  number_search_term_conjunctions[i] = length(search_term_conjunctions[[i]])
  number_search_term_numerals[i] = length(search_term_numerals[[i]])
  
  #Determine the number of words in product_title for each part of speech
  number_product_title_nouns[i] = length(product_title_nouns[[i]])
  number_product_title_verbs[i] = length(product_title_verbs[[i]])
  number_product_title_adjectives[i] = length(product_title_adjectives[[i]])
  number_product_title_adverbs[i] = length(product_title_adverbs[[i]])
  number_product_title_conjunctions[i] = length(product_title_conjunctions[[i]])
  number_product_title_numerals[i] = length(product_title_numerals[[i]])
  
  #Determine the number of matching words between search_term and product_title for each part of speech
  number_match_title_nouns[i] = matching_number(search_term_nouns[[i]], product_title_nouns[[i]])
  number_match_title_verbs[i] = matching_number(search_term_verbs[[i]], product_title_verbs[[i]])
  number_match_title_adjectives[i] = matching_number(search_term_adjectives[[i]], product_title_adjectives[[i]])
  number_match_title_adverbs[i] = matching_number(search_term_adverbs[[i]], product_title_adverbs[[i]])
  number_match_title_conjunctions[i] = matching_number(search_term_conjunctions[[i]], product_title_conjunctions[[i]])
  number_match_title_numerals[i] = matching_number(search_term_numerals[[i]], product_title_numerals[[i]])
  
  #Determine the ratio of matching words between search_term and product_title for each part of speech
  ratio_match_title_nouns[i] = matching_ratio(search_term_nouns[[i]], product_title_nouns[[i]])
  ratio_match_title_verbs[i] = matching_ratio(search_term_verbs[[i]], product_title_verbs[[i]])
  ratio_match_title_adjectives[i] = matching_ratio(search_term_adjectives[[i]], product_title_adjectives[[i]])
  ratio_match_title_adverbs[i] = matching_ratio(search_term_adverbs[[i]], product_title_adverbs[[i]])
  ratio_match_title_conjunctions[i] = matching_ratio(search_term_conjunctions[[i]], product_title_conjunctions[[i]])
  ratio_match_title_numerals[i] = matching_ratio(search_term_numerals[[i]], product_title_numerals[[i]])
  
}

#Add the above vectors to data
data[, number_search_term_nouns := number_search_term_nouns]
data[, number_search_term_verbs := number_search_term_verbs]
data[, number_search_term_adjectives := number_search_term_adjectives]
data[, number_search_term_adverbs := number_search_term_adverbs]
data[, number_search_term_conjunctions := number_search_term_conjunctions]
data[, number_search_term_numerals := number_search_term_numerals]

data[, number_product_title_nouns := number_product_title_nouns]
data[, number_product_title_verbs := number_product_title_verbs]
data[, number_product_title_adjectives := number_product_title_adjectives]
data[, number_product_title_adverbs := number_product_title_adverbs]
data[, number_product_title_nouns := number_product_title_nouns]
data[, number_product_title_conjunctions := number_product_title_conjunctions]
data[, number_product_title_numerals := number_product_title_numerals]

data[, number_match_title_nouns := number_match_title_nouns]
data[, number_match_title_verbs := number_match_title_verbs]
data[, number_match_title_adjectives := number_match_title_adjectives]
data[, number_match_title_adverbs := number_match_title_adverbs]
data[, number_match_title_conjunctions := number_match_title_conjunctions]
data[, number_match_title_numerals := number_match_title_numerals]

data[, ratio_match_title_nouns := ratio_match_title_nouns]
data[, ratio_match_title_verbs := ratio_match_title_verbs]
data[, ratio_match_title_adjectives := ratio_match_title_adjectives]
data[, ratio_match_title_adverbs := ratio_match_title_adverbs]
data[, ratio_match_title_conjunctions := ratio_match_title_conjunctions]
data[, ratio_match_title_numerals := number_product_title_numerals]

#Calculate various distance metrics for the nouns in search_term and product_title

#Initialise Vectors
search_title_cosine_nouns = vector(mode="numeric", length=nrow(data))
search_title_jaccard_nouns = vector(mode="numeric", length=nrow(data))
search_title_jw_nouns = vector(mode="numeric", length=nrow(data))

for (i in 1:nrow(data)) {
  print(i)
  
  search_term_nouns[i] = paste(search_term_nouns[[i]], collapse=" ")
  product_title_nouns[i] = paste(product_title_nouns[[i]], collapse=" ")
  
}

search_term_nouns = unlist(search_term_nouns)
product_title_nouns = unlist(product_title_nouns)

#Calculate pairwise string distances for nouns
data[, search_title_cosine_nouns := stringdist(search_term_nouns, product_title_nouns, method="cosine")]
data[, search_title_jaccard_nouns := stringdist(search_term_nouns, product_title_nouns, method="jaccard")]
data[, search_title_jw_nouns := stringdist(search_term_nouns, product_title_nouns, method="jw")]

#Replace infitite distance measures with missing
data[is.infinite(search_title_cosine_nouns), search_title_cosine_nouns := NA]
data[is.infinite(search_title_jaccard_nouns), search_title_jaccard_nouns := NA]
data[is.infinite(search_title_jw_nouns), search_title_jw_nouns := NA]

#Extract categories relating to colour and brand in attribute names
colour_names = unique(tolower(attributes$name[grepl("color", tolower(attributes$name))]))
brand_names = unique(tolower(attributes$name[grepl("brand", tolower(attributes$name))]))

#Create dummy variables relating to value in colour_names
for (colour_name in colour_names) {
  print(colour_name)
  VarName = paste("has", make.names(colour_name), sep="_")
  data[, (VarName) := as.numeric(grepl(paste(". ", colour_name, ".", sep=""), name, fixed=T))]
}

#Create dummy variables relating to value in brand_names
for (brand_name in brand_names) {
  print(brand_name)
  VarName = paste("has", make.names(brand_name), sep="_")
  data[, (VarName) := as.numeric(grepl(paste(". ", brand_name, ".", sep=""), name, fixed=T))]
}

#Create dummy variable indicating whether a description of the material is provided for each row
data[, has_material := as.numeric(grepl(". material.", name, fixed=T))]

#Count the number of dummy variables by colour and brand
data[, count_colour := rowSums(((data[, grepl("color", names(data)), with = F])))]
data[, count_brand := rowSums(((data[, grepl("brand", names(data)), with = F])))]

#Word stem
cols = c("product_description", "product_title", "search_term", "value")
data[, (cols) := lapply(.SD, stemming), .SDcols = cols]

#Create a vector of possible colours
colours = attributes[name == "Color" | name == "Color Family", value]

#Clean strings
colours = unique(clean_string(colours))

#Select only the rows of attributes where name = Color
attributes_colour = attributes[name == "Color"]
attributes_colour[, value := clean_string(value)]

#Find colours with low frequency and remove them from vector of possible colours
remove_colours = attributes_colour %>%
  select(value) %>%
  group_by(value) %>%
  summarise(colour_freq = n()) %>%
  filter(colour_freq < 4) %>%
  .$value

#Remove all colours contained in remove_colours
colours = setdiff(colours, remove_colours)

#Stem colours
colours = stemming(colours)

#Initiliase empty vectors of appropriate type and length
search_term_colours = vector(mode="character", length=nrow(data))
product_title_colours = vector(mode="character", length=nrow(data))
product_description_colours = vector(mode="character", length=nrow(data))
product_attributes_value_colours = vector(mode="character", length=nrow(data))

#Loop through the vector of colours, and if a colour exists in search_term, product_title or value, concatenate that colour to the new
#variable

cols = c("search_term", "product_title", "value")
cols_new = paste(cols, "colours", sep="_")

data[, (cols_new) := lapply(.SD, match_variable, colours), .SDcols = cols]

#Word stem colour variables
cols = c("search_term_colours", "product_title_colours", "value_colours")
data[, (cols) := lapply(.SD, stemming), .SDcols = cols]

#Split colour strings
search_term_colour_split = strsplit(data$search_term_colours, " . ")
product_title_colour_split = strsplit(data$product_title_colours, " . ")
value_colour_split = strsplit(data$value_colours, " . ")

#Clean and trim strings
search_term_colour_split = lapply(search_term_colour_split, clean_trim_string)
product_title_colour_split = lapply(product_title_colour_split, clean_trim_string)
value_colour_split = lapply(value_colour_split, clean_trim_string)

#Initialise empty vector with the appropriate type and lenth
ratio_colour_match_title = vector(mode="numeric", length=nrow(data))
ratio_colour_match_attributes = vector(mode="numeric", length=nrow(data))

for (i in 1:nrow(data)) {
  print(i)
  if (length(search_term_colour_split[[i]]) > 0) {
    #Assess ratio of matching colour words between search_term and product_title and product_description
    ratio_colour_match_title[i] = matching_ratio(search_term_colour_split[[i]], product_title_colour_split[[i]])
    ratio_colour_match_attributes[i] = matching_ratio(search_term_colour_split[[i]], value_colour_split[[i]])
  } else {
    ratio_colour_match_title[i] = NA
    ratio_colour_match_attributes[i] = NA
  }
}

#Add the vectors to data
data[, ratio_colour_match_title := ratio_colour_match_title]
data[, ratio_colour_match_attributes := ratio_colour_match_attributes]

#Remove unescessary variables from data
data[, ':=' (search_term_colours = NULL,
             product_title_colours = NULL,
             value_colours = NULL)]

#Creating features from brand information

#Obtain possible brands
attributes_brands = attributes[name == "MFG Brand Name", !"name", with = F]

#Change the name of the column value to brand
setnames(attributes_brands, "value", "brand")

#Clean the brands
attributes_brands[, brand := clean_string(brand)]

#Word stem brands
attributes_brands[, brand := stemming(brand)]

#Create ranking of brands based on the number of times each product is returned from a search
rank_brands = attributes_brands %>%
  select(brand) %>%
  group_by(brand) %>%
  summarise(brand_freq = n()) %>%
  arrange(desc(brand_freq)) %>%
  mutate(rank_brand = 1:length(brand)) %>%
  select(brand, rank_brand)

#Set appropriate keys prior to joining
setkey(attributes_brands, brand)
setkey(rank_brands, brand)

#Join attributes_brands with rank_brand
attributes_brands = rank_brands[attributes_brands]

#Set appropriate keys prior to joining
setkey(data, product_uid)
setkey(attributes_brands, product_uid)

#Join data with attributes_brands
data = attributes_brands[data]

#If brand is missing - set it to an empty string
data[is.na(brand), brand := ""]

#Check whether search term matches any brand
#Initialise empty vector of appropriate type and length
search_term_has_brand = vector(mode="numeric", length=nrow(data))

#Identify which search terms contain a brand
unique_brands = attributes_brands[, unique(brand)]
for (unique_brand in unique_brands) {
  print(unique_brand)
  matches = which(grepl(unique_brand, data$search_term, fixed=T))
  search_term_has_brand[matches] = 1
}

#Add the above vector to data
data[, search_term_has_brand := search_term_has_brand]

#Feature to account for the different matching permutations by brand between search_term and product brand
#grepl requires a single pattern (not a patern vector). Therefore need to loop through each row.
for (i in 1:nrow(data)) {
  print(i)
  data[i, search_term_brand_info := feature_permutations_1(search_term_has_brand, brand, search_term)]
}

#Home Depot's relevance rules state that if a user only searches for a brand, then the relevance score should be 3 if any product of that
#brand is returned 
#Create a feature to account for this rule
data[, search_term_brand_equal := as.numeric(search_term == brand)]

#Split strings into lists of words
search_term_split = strsplit(data$search_term, " ")
brand_split = strsplit(data$brand, " ")

#Initialise Vector of appropriate type and length
ratio_search_term_match_brand = vector(mode="numeric", length=nrow(data))

#Determine ratio of matching brand words in the search term and the product returned
for (i in 1:nrow(data)) {
  print(i)
  if (data$brand[i] != "") {
    ratio_search_term_match_brand[i] = matching_ratio(brand_split[[i]], search_term_split[[i]])
  } else {
    ratio_search_term_match_brand[i] = NA
  }
}

#Add the above vector to data
data[, ratio_search_term_match_brand := ratio_search_term_match_brand]

#Split strings into lists of words
search_term_split = strsplit(data$search_term, " ")
product_title_split = strsplit(data$product_title, " ")
product_description_split = strsplit(data$product_description, " ")
attributes_split = strsplit(data$value, " ")
brand_split = strsplit(data$brand, " ")

#Create vector of possible colours
colours = unique(attributes$value[grepl("Color", attributes$name)])
colours = clean_string(colours)

#Word stem colours
colours = stemming(colours)

#Count number of words in search, product_title, product_description and brand
data[, search_num_words := sapply(search_term_split, length)]
data[, product_title_num_words := sapply(product_title_split, length)]
data[, product_description_num_words := sapply(product_description_split, length)]
data[, brand_num_words := sapply(brand_split, length)]

#Count number of charcters in search, product_title and product_description
data[, search_num_characters := sapply(search_term_split, count_chars)]
data[, product_title_num_characters := sapply(product_title_split, count_chars)]
data[, product_description_num_characters := sapply(product_description_split, count_chars)]

#Count number of charcters characters per word
data[, search_avg_characters := search_num_characters/search_num_words]
data[, product_title_avg_characters := product_title_num_characters/product_title_num_words]
data[, product_description_avg_characters := product_description_num_characters/product_description_num_words]

#Initialise vectors of appropriate type and length
match_title = vector(mode="numeric", length=nrow(data))
match_description = vector(mode="numeric", length=nrow(data))

number_match_title = vector(mode="numeric", length=nrow(data))
number_match_description = vector(mode="numeric", length=nrow(data))
number_match_attributes = vector(mode="numeric", length=nrow(data))

ratio_match_title = vector(mode="numeric", length=nrow(data))
ratio_match_description = vector(mode="numeric", length=nrow(data))
ratio_match_attributes = vector(mode="numeric", length=nrow(data))

search_colours = vector(mode="numeric", length=nrow(data))
product_title_colours = vector(mode="numeric", length=nrow(data))
product_description_colours = vector(mode="numeric", length=nrow(data))

title_colour_info = vector(mode="character", length=nrow(data))
description_colour_info = vector(mode="character", length=nrow(data))

#Loop through each row in data
for (i in 1:nrow(data)) {
  
  print(i)
  
  #Assess whether there is an exact match (in order) between query and title and/or description
  match_title[i] = exact_match(data$search_term[i], data$product_title[i])
  match_description[i] = exact_match(data$search_term[i], data$product_description[i])

  #Assess number of matching words between query and title, description and attributes
  number_match_title[i] = matching_number(search_term_split[[i]], product_title_split[[i]])
  number_match_description[i] = matching_number(search_term_split[[i]], product_description_split[[i]])
  number_match_attributes[i] = matching_number(search_term_split[[i]], attributes_split[[i]])
  
  #Assess ratio of matching words between query and title, description and attributes
  ratio_match_title[i] = matching_ratio(search_term_split[[i]], product_title_split[[i]])
  ratio_match_description[i] = matching_ratio(search_term_split[[i]], product_description_split[[i]])
  ratio_match_attributes[i] = matching_ratio(search_term_split[[i]], attributes_split[[i]])
  
  #Colour Matching
  search_colours[i] = intersection_values(search_term_split[[i]], colours)
  product_title_colours[i] = intersection_values(product_title_split[[i]], colours)
  product_description_colours[i] = intersection_values(product_description_split[[i]], colours)
  
  title_colour_info[i] = feature_permutations_2(search_colours[i], product_title_colours[i])
  description_colour_info[i] = feature_permutations_2(search_colours[i], product_description_colours[i])
  
}

#Add variables to data
data[, match_title := match_title]
data[, match_description := match_description]

data[, number_match_title := number_match_title]
data[, number_match_description := number_match_description]
data[, number_match_attributes := number_match_attributes]

data[, ratio_match_title := ratio_match_title]
data[, ratio_match_description := ratio_match_description]
data[, ratio_match_attributes := ratio_match_attributes]

data[, title_colour_info := title_colour_info]
data[, description_colour_info := description_colour_info]

#Calculate pairwise string distances

data[, search_title_jaccard := stringdist(search_term, product_title, method="jaccard")]
data[, search_title_osa := stringdist(search_term, product_title, method="osa")]
data[, search_title_lv := stringdist(search_term, product_title, method="lv")]
data[, search_title_dl := stringdist(search_term, product_title, method="dl")]
data[, search_title_lcs := stringdist(search_term, product_title, method="lcs")]
data[, search_title_qgram := stringdist(search_term, product_title, method="qgram")]
data[, search_title_cosine := stringdist(search_term, product_title, method="cosine")]
data[, search_title_jw := stringdist(search_term, product_title, method="jw")]
data[, search_title_soundex := stringdist(search_term, product_title, method="soundex")]

data[, search_description_jaccard := stringdist(search_term, product_description, method="jaccard")]
data[, search_description_osa := stringdist(search_term, product_description, method="osa")]
data[, search_description_lv := stringdist(search_term, product_description, method="lv")]
data[, search_description_dl := stringdist(search_term, product_description, method="dl")]
data[, search_description_lcs := stringdist(search_term, product_description, method="lcs")]
data[, search_description_qgram := stringdist(search_term, product_description, method="qgram")]
data[, search_description_cosine := stringdist(search_term, product_description, method="cosine")]
data[, search_description_jw := stringdist(search_term, product_description, method="jw")]
data[, search_description_soundex := stringdist(search_term, product_description, method="soundex")]

#Count number of times each product_uid occurs in data
count_product_uids = data %>%
  select(product_uid) %>%
  group_by(product_uid) %>%
  summarise(count_product_uid = n())

#Set appropriate keys prior to joining
setkey(data, product_uid)
setkey(count_product_uids, product_uid)
setkey(count_product_uids_train, product_uid)

#Join data with count_product_uids and count_product_uids_train
data = count_product_uids[data]
data = count_product_uids_train[data]

#Count number of times each search_term occurs in data
count_search_terms = data %>%
  select(search_term) %>%
  group_by(search_term) %>%
  summarise(count_search_term = n())

#Set appropriate keys prior to joining
setkey(data, search_term)
setkey(count_search_terms, search_term)
setkey(count_search_terms_train, search_term)

#Join data with count_search_terms and count_search_terms_train
data = count_search_terms[data]
data = count_search_terms_train[data]

#Remove raw text features no longer required
data[, ':=' (product_title = NULL,
             product_description = NULL,
             brand = NULL,
             name = NULL,
             value = NULL)]

#One-hot-encode factors (currently encoded as character variables)

#Names of character columns to be one-hot-encoded
character_names = c("search_term_brand_info", "title_colour_info", "description_colour_info")

#One-hot-encode data.table by selecting all columns except for those in character_names and cbind the result with a matrix applied to
#the character columns
data = cbind(data[, !character_names, with = F], model.matrix(~ search_term_brand_info + title_colour_info + description_colour_info - 1, data))

#Make the column names suitable for R
names(data) = make.names(names(data), unique = T)

#Split data back into train and test
train = data[!is.na(relevance)]
test = data[is.na(relevance)]

#Apply leave-one-out encoding to data - necessary becaause search_term and product_uid are both high cardinality factors
LOO = LooEncoding(train, test, c("search_term", "product_uid"), "relevance", "id")

#Can now remove search_term
train[, search_term := NULL]
test[, search_term := NULL]

#Set appropriate keys prior to joining
setkey(train, id)
setkey(test, id)
setkey(LOO, id)

#Join train and test with LOO
train = LOO[train]
test = LOO[test]

#Add overlap indicator variables
train[, search_term_overlap := ifelse(is.na(search_term_relevance), 0, 1)]
test[, search_term_overlap := ifelse(is.na(search_term_relevance), 0, 1)]

train[, product_overlap := ifelse(is.na(product_uid_relevance), 0, 1)]
test[, product_overlap := ifelse(is.na(product_uid_relevance), 0, 1)]

#Read in regularised linear model predictions on the tf-idf
load("glm_tf_idf.RData")

#Set appropriate keys prior to joining
setkey(train_glm_tf_idf, id)
setkey(test_glm_tf_idf, id)

#Join train and test with the model predictions
train = train_glm_tf_idf[train]
test = test_glm_tf_idf[test]

#Load ngram features
load("ngram_data.RData")

#Set appropriate key prior to joining
setkey(ngram_data, id)

#Join train and test with ngram_data
train = ngram_data[train]
test = ngram_data[test]

#Remove relevance from test - only a placeholder
test[, relevance := NULL]

#Remove any zero variance columns
trainZeroVar = names(train[, sapply(train, function(v) var(v, na.rm=TRUE) == 0), with = F])
#Ensure that id and relevance are not removed - these are not features and so should not be removed from feature set
trainZeroVar = trainZeroVar[!trainZeroVar %in% c("id", "relevance")]
for (f in trainZeroVar) {
  train[, (f) := NULL]
  test[, (f) := NULL]
}

#Remove highly correlated features
trainCor = cor(train, use = "complete.obs")
trainCor[is.na(trainCor)] = 0
trainCor[upper.tri(trainCor)] = 0
diag(trainCor) = 0

trainRemove = names(train[,apply(trainCor,2,function(x) any(x >= 0.999 | x <= -0.999)), with = F])

#Ensure that id and relevance are not removed - these are not features and so should not be removed from feature set
trainRemove = trainRemove[!trainRemove %in% c("id", "relevance")]

train[, (trainRemove) := NULL]
test[, (trainRemove) := NULL]

#Save train and test as RData
#The resulting R objects are used as feature sets for various stage 0 models within the stacked generalisation
save(list = c("train", "test"), file = "feature_engineering_set_1.RData")

#Create another feature set based on feature_engineering_set_1 - select only the features deemed most significant in a gradient boosted
#decision tree, that is, the features with the highest gain contributions to the model
train_keep_cols = c("id", "relevance", "ratio_match_title", "product_title_ngram_simscore", "search_term_relevance", "count_search_term", 
              "product_description_ngram_matchscore", "product_title_1grams", "glm_pred", "product_title_ngram_matchscore",
              "ratio_match_description", "search_num_characters", "search_title_jaccard",
              "ratio_match_title_nouns", "search_title_cosine_nouns", "search_term_overlap")

test_keep_cols = train_keep_cols[!train_keep_cols %in% "relevance"]

#Create train and testing sets for the reduced feature set based on the columns above
train_reduced = train[, train_keep_cols, with = F]
test_reduced = test[, test_keep_cols, with = F]

#Add indicator variables (may be helpful in linear models)
train_reduced[, ratio_match_title_nouns_dummy := ifelse(is.na(ratio_match_title_nouns), 1, 0)]
test_reduced[, ratio_match_title_nouns_dummy := ifelse(is.na(ratio_match_title_nouns), 1, 0)]

train_reduced[, search_title_cosine_nouns_dummy := ifelse(is.na(search_title_cosine_nouns), 1, 0)]
test_reduced[, search_title_cosine_nouns_dummy := ifelse(is.na(search_title_cosine_nouns), 1, 0)]

#Save train_reduced and test_reduced as RData
save(list = c("train_reduced", "test_reduced"), file = "feature_engineering_set_1_reduced.RData")

#Load feature_engineering tf_idf
#Replace the regularised linear model predictions in feature_set_1 with the data used to obtain those predictions (tf_idf) in order to
#create a different feature set for various stage 0 models within the stacked generalisation
load("feature_engineering_tf_idf.RData")

#Remove linear model predictions - using tf_idf as features instead
train[, glm_pred := NULL]
test[, glm_pred := NULL]

#Set appropriate keys prior to joining
setkey(train_tf_idf, id)
setkey(test_tf_idf, id)

#Join train and test with train_tf_idf and test_tf_idf
train = train_tf_idf[train]
test = test_tf_idf[test]

#Save train and test as RData
save(list = c("train", "test"), file = "feature_engineering_set_2.RData")

#Detach all packages from session to avoid conficts with other scripts
detachAllPackages()

#Remove all objects from the workspace
rm(list = ls())
gc()

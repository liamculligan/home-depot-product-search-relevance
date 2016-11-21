#Home Depot Product Search Relevance

#Functions

#Authors: Tyrone Cragg & Liam Culligan

#Date: April 2016

#Clean product_description and title
#If words are concatenated incorrectly, e.g. aA, separate to a A
split_text = function(x) {
  return(gsub("([a-z])([A-Z])([a-z])", "\\1 \\2\\3", x))
}

#Correct spelling in input vector
correct_spelling = function(original, incorrect, correct) {
  for (i in 1:length(incorrect)) { 
    original[original == incorrect[i]] = correct[i]
  }
  return(original)
}

#Remove punctuation and whitespaces from character variables
remove_punctuation_whitespace = function(x) {
  x = gsub("[[:punct:]]", " ", x)
  x = gsub("\\s+", " ", x)
  return(x)
}

#Identify where units measurements exist and to standardise these measurements
#e.g. 6 feet becomes 6ft
standardise_units = function(x, unit, unit_standard) {
  x = gsub(paste("([0-9]+)( *)(", unit, ")", sep=""), paste("\\1", unit_standard, sep=""), x)
  x = gsub(paste("([0-9]+)(-)(", unit, ")", sep=""), paste("\\1", unit_standard, sep=""), x)
  x = gsub(paste("([0-9]+)(", unit, ")", sep=""), paste("\\1", unit_standard, sep=""), x)
  return(x)
}

#Similar function to standardise measurements
standardise_measurements = function(x) {
  x = gsub("doe cu ft", "doe cuft", x)
  x = gsub("([0-9]+)(x)([0-9]+)", "\\1 xby \\3", x)
  x = gsub("([0-9]+)(x)( *)([0-9]+)", "\\1 xby \\4", x)
  x = gsub("([0-9]+)( *)(x)([0-9]+)", "\\1 xby \\4", x)
  return(x)
}

#Clean strings
clean_string = function(x) {
  x = tolower(x)
  x = gsub("[[:punct:]]", " ", x)
  x = gsub("\\s+", " ", x)
}

#Similar function to above
clean_trim_string = function(x) {
  library(stringr)
  x = gsub("[[:punct:]]", " ", x)
  x = gsub("\\s+", " ", x)
  x = str_trim(x, "both")
  return(x)
}

#Loop through a vector of strings, and if a string exists in a row, concatenate that string to the new variable for that row
match_variable = function(x, matches) {
  
  matches_string = vector(mode="character", length=length(x))
  
  for (match in matches) {
    print(match)
    rows = which(grepl(paste("^", match, " ", sep = ""),  x) |
                   grepl(paste(" ", match, "$", sep = ""),  x) |
                   grepl(paste(" ", match, " ", sep = ""),  x))
    matches_string[rows] = paste(matches_string[rows], " ", match, " .", sep="")
  }
  return(matches_string)
}

#Determine the number of matching strings in 2 lists of words
matching_number = function(x, y) {
  return(length(intersect(x, y)))
}

#Determine the ratio of matching strings in 2 character vectors
matching_ratio = function(x, y) {
  return(length(intersect(x, y))/length(x))
}

#Word stemming multiple words per row
stemming = function(x) {
  library(tm)
  corpus = Corpus(VectorSource(x))
  map = tm_map(corpus, stemDocument)
  x = sapply(map, "[[", 1)
  return(x)
}

#Identify the permutations regarding a feature in 2 variables
feature_permutations_1 = function(x, y, z) {
  feature_info = ifelse(x == 0 & y == "", "Not present in X. Not present in Y", 
                        ifelse(x == 0 & y != "", "Not present in X. Present in Y.",
                               ifelse(x == 1 & y != "" & grepl(y, z) == T, "Same in X and Y",
                                      ifelse(x == 1 & y != "" & grepl(y, z) == F, "Different in X and Y",
                                             ifelse(x == 1 & y == "", "Present in X. Not present in Y", "")))))
  return(feature_info)
}

#Similar to above. Create function to identify the permutations regarding a feature in 2 variables
feature_permutations_2 = function(x, y, z) {
  feature_info = ifelse(x == "" & y == "", "Not present in X. Not present in Y", 
                        ifelse(x == "" & y != "", "Not present in X. Present in Y.",
                               ifelse(x != "" & length(intersect(x, y)) > 0, "Same in X and Y",
                                      ifelse(x != "" & y != "" & length(intersect(x, y)) == 0, "Different in X and Y",
                                             ifelse(x != "" & y == "", "Present in X. Not present in Y", "")))))
  return(feature_info)
}

#Count number of characters
count_chars = function(x) {
  return(sum(nchar(x)))
}

#Determine if exact match in order exists between 2 strings
exact_match = function(x, y) {
  return(ifelse(length(grep(x, y)) > 0, 1, 0))
}

#Function to return intersection values between two vectors
intersection_values = function(x, y) {
  return(ifelse(length(intersect(x, y)) > 0, intersect(x, y), ""))
}

#Leave-one-out encoding 
LooEncoding = function(train, test, factorVec, targetName, idName) {
  #factorName, targetName and idName provided as strings
  library(data.table)
  data = data.table(id = c(train[[idName]], test[[idName]]))
  names(data) = idName
  
  setkeyv(data, idName)
  
  for (factorName in factorVec) { 
    
    LOO = train[, .(meanTarget = unlist(lapply(.SD, mean)), n = unlist(lapply(.SD, length))),
                .SDcols = targetName, by = factorName]
    
    setkeyv(LOO, factorName)
    setkeyv(train, factorName)
    
    trainTemp = LOO[train]
    
    #Add randomness
    set.seed(44)
    trainRand = runif(nrow(trainTemp), 0.95, 1.05)
    
    trainTemp[, meanTarget := (((meanTarget * n) - trainTemp[[targetName]])/(n-1)) * trainRand] 
    
    setkeyv(test, factorName)
    
    testTemp = LOO[test]
    
    dataTemp = data.table(id=c(trainTemp[[idName]], testTemp[[idName]]),
                          var=c(trainTemp$meanTarget, testTemp$meanTarget))
    names(dataTemp) = c(idName, paste(factorName, targetName, sep="_"))
    
    setkeyv(dataTemp, idName)
    
    data = dataTemp[data]
    
  }
  
  return(data)
  #Returns data.frame of ids and the result of LOO Hot Encoding. Ready to merge with train and test by id.
  
}

#For each search term, determine all possible n-grams (up to 4)

#Then create a number of similarity metrics between the search term and both product_title and product_description:
#word lengths, number of matches for each n-gram with product_title, number matches for each n-gram with
#product_descrtiption, ratio of n-grams found in product_title, ratio of n-grams found in product_descrtiption,
#average similarity score for all n-grams with product_title, average similarity score for all n-grams with
#product_descrtiption, average similarity score for shuffled bigrams with product_title, average similarity
#score for shuffled bigrams with product_descrtiption, similarity score for full search term with product_title
#and similarity score for full search term with product_descrtiption

ngramMatches = function(pattern, string1, string2) {
  #Pattern is a search term, string1 and string2 are product title and product description 
  
  library(RWeka)
  library(stringdist)
  library(combinat)
  
  #Extract number of words in search term, product title and product description
  
  #Split by each word in search term
  w1 = NGramTokenizer(pattern, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  #Number of words
  n1 = length(w1)
  
  w2 = NGramTokenizer(string1, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  n2 = length(w2)
  w3 = NGramTokenizer(string2, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  n3 = length(w3)
  
  #Add lengths to output
  out = c(n1, n2, n3)
  
  #Extract all n-grams in search term, up to 4 or the number of words in search term (if less than 4)
  w1 = NGramTokenizer(pattern, Weka_control(min = 1, max = min(n1,4), delimiters = " \\r\\n.?!:"))
  
  #Number of words in each gram (count number of spaces and add one)
  w1.gramLen = nchar(gsub("[^ ]", "", w1)) + 1
  
  #Initialise vectors
  rslt1 = rep(0,4)
  rslt2 = rep(0,4)
  r1 = rep(0,length(w1))
  r2 = rep(0,length(w1))
  r3 = rep(0,length(w1))
  r4 = rep(0,length(w1))
  
  #Loop through each gram in search term
  for (i in 1:length(w1)) {
    
    #For this length of n-gram, apply ntimes to add 1 if n-gram appears in product_title
    rslt1[w1.gramLen[i]] = rslt1[w1.gramLen[i]] + ntimes(w1[i], string1)
    
    #For this length of n-gram, apply ntimes to add 1 if n-gram appears in product_descrtiption
    rslt2[w1.gramLen[i]] = rslt2[w1.gramLen[i]]  + ntimes(w1[i], string2)
    
    #Binary variable indicating whether this n-gram occurs in product_title
    r1[i] = ifelse(ntimes(w1[i], string1) > 0, 1, 0)
    
    #Binary variable indicating whether this n-gram occurs in product_descrtiption
    r2[i] = ifelse(ntimes(w1[i], string2) > 0, 1, 0)
    
    #Similarity measure for this n-gram and product_title
    r3[i] = simscore(w1[i], string1)
    
    #Similarity measure for this n-gram and product_descrtiption
    r4[i] = simscore(w1[i], string2)
  } 
  
  #Add to outputs
  #Number of matches for each n-gram with product_title,
  #number of matches for each n-gram with product_descrtiption,
  #ratio of n-grams found in product_title,
  #ratio of n-grams found in product_descrtiption,
  #average similarity score for n-grams with product_title,
  #average similarity score for n-grams with product_descrtiption
  out = c(out, rslt1, rslt2, sum(r1)/length(r1), sum(r2)/length(r2), sum(r3)/length(r3), sum(r4)/length(r4))
  
  #Second order bigram matching (reshuffle words in query)
  
  #Extract unigrams from search term
  w1.single = w1[w1.gramLen==1]
  
  #Extract bigrams from search term
  w1.bigram = w1[w1.gramLen==2]
  
  #Use combinat package to create all possible bigrams by shuffling words
  all.comb = apply(rbind(combn2(w1.single), combn2(w1.single)[,2:1]), 1, paste, collapse =" ")
  
  #Remove bigrams that already exist
  all.comb = all.comb[!all.comb %in% w1.bigram]
  
  #Initialise vectors
  r5 = rep(0,length(all.comb))
  r6 = rep(0,length(all.comb))
  
  #Loop through new bigrams
  for (i in 1:length(all.comb)) {
    
    #Similarity measure for this bigram and product_title
    r5[i] = simscore(all.comb[i], string1)
    
    #Similarity measure for this bigram and product_description
    r6[i] = simscore(all.comb[i], string2)
  }
  
  #Add to outputs
  #Average similarity score for new bigrams with product_title,
  #average similarity score for new bigrams with product_descrtiption
  out = c(out, sum(r5)/length(r5), sum(r6)/length(r6))
  
  #Add to outputs
  #Similarity score for full search term with product_title,
  #Similarity score for full search term with product_descrtiption
  out = c(out, simscore(pattern, string1), simscore(pattern, string2))
  
  return(out)
}

#Calculate similarity scroe between two strings - applying longest common subsequence
simscore = function(pattern, string) {
  1-pmin(stringdist( pattern, string, method="lcs") - max(nchar(string) - nchar(pattern), 0),nchar(pattern))/nchar(pattern)
}

#Number of characters in string2 minus number of characters in string2 after string1 has been removed from string2,
#all divided by number of characters in string1
#Will equal to 1 if string1 appears in string2, otherwise 0
ntimes = function (string1, string2){
  string1 = gsub("[ -]", "[ ]?", string1)
  return((nchar(string2) - nchar(gsub(paste("", string1, "", sep=""), "", string2)))/nchar(string1))
}

#Detach all packages from session
detachAllPackages = function() {
  basic.packages = c("package:stats","package:graphics","package:grDevices","package:utils","package:datasets","package:methods","package:base")
  package.list = search()[ifelse(unlist(gregexpr("package:",search()))==1,TRUE,FALSE)]
  package.list = setdiff(package.list,basic.packages)
  if (length(package.list)>0)  for (package in package.list) detach(package, character.only=TRUE)
}

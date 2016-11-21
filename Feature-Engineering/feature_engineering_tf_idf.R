#Home Depot Product Search Relevance

#Feature engineering to create term frequency–inverse document frequency

#Authors: Tyrone Cragg & Liam Culligan

#Date: April 2016

#Load required packages
library(data.table)
library(tm)
library(text2vec)
library(RSpectra)
library(Matrix)

#Read in required functions
source("functions.R") 

#Load pre-processed data
load("pre_process.RData")

#Don't require the data.table attributes to create tf_idf - remove this data.table
rm(attributes)
gc()

#Don't require the variables name, value or product_uid to create tf_idf - remove from data
data[, ':=' (name = NULL,
             value = NULL,
             product_uid = NULL)]

#Word stemming
stemming = function(x) {
  library(tm)
  corpus = Corpus(VectorSource(x))
  map = tm_map(corpus, stemDocument)
  x = sapply(map, "[[", 1)
  return(x)
}

data[, (cols) := lapply(.SD, stemming), .SDcols = cols]

#Convert search_term to tf-idf

#Create vocabulary
#Each element of list represents a document
tokens = data$search_term %>% 
  word_tokenizer()

it = itoken(tokens, ids = data$id)
vocab = create_vocabulary(it)

#Now that we have a vocabulary, we can construct a document-term matrix
it = itoken(tokens, ids = data$id)
vectorizer = vocab_vectorizer(vocab)
dtm = create_dtm(it, vectorizer)

#Now we have a DTM and can check its dimensions.
str(dtm)

#Pruning vocabulary - remove both very common and very unusual terms
pruned_vocab = prune_vocabulary(vocab, term_count_min = 10, 
                                doc_proportion_max = 0.5, doc_proportion_min = 0.003)
it = itoken(tokens, ids = data$id)
vectorizer = vocab_vectorizer(pruned_vocab)
dtm = create_dtm(it, vectorizer)

#Check dimensions of reduced dtm
dim(dtm)

#Apply TF-IDF transformation to our DTM which will increase the weight of terms which are specific to
#a single document or handful of documents and decrease the weight for terms used in most documents
dtm = dtm %>%
  transform_tfidf()

#Find the Largest 100 Singular Values/Vectors of the dtm - reduced the number of features
k = 100
svds = svds(dtm, k = k)
m_truncated = dtm %*% svds$v

#Convert dtm to the reduced feature set
dtm = as.matrix(m_truncated)
colnames(dtm) = 1:k
colnames(dtm) = paste("search",colnames(dtm), sep="_")
data = cbind(data, dtm)

#Convert product_title to tf-idf

#Create Vocabulary
#Each element of list represents document
tokens = data$product_title %>% 
  word_tokenizer()

it = itoken(tokens, ids = data$id)
vocab = create_vocabulary(it)

#Now that we have a vocabulary, we can construct a document-term matrix
it = itoken(tokens, ids = data$id)
vectorizer = vocab_vectorizer(vocab)
dtm = create_dtm(it, vectorizer)

#Now we have a DTM and can check its dimensions.
str(dtm)

#Pruning vocabulary - remove both very common and very unusual terms
pruned_vocab = prune_vocabulary(vocab, term_count_min = 10, doc_proportion_max = 0.5, doc_proportion_min = 0.009)

it = itoken(tokens, ids = data$id)
vectorizer = vocab_vectorizer(pruned_vocab)
dtm = create_dtm(it, vectorizer)

#Check dimensions of reduced dtm
dim(dtm)

#Apply TF-IDF transformation to our DTM which will increase the weight of terms which are specific to
#a single document or handful of documents and decrease the weight for terms used in most documents
dtm = dtm %>% transform_tfidf()

#Find the Largest 100 Singular Values/Vectors of the dtm - reduced the number of features
k = 100
svds = svds(dtm, k = k)
m_truncated = dtm %*% svds$v

#Convert dtm to the reduced feature set
dtm = as.matrix(m_truncated)
colnames(dtm) = 1:k
colnames(dtm) = paste("title",colnames(dtm), sep="_")
data = cbind(data, dtm)

#Convert product_description to tf-idf

#Create Vocabulary
#Each element of list represents document
tokens = data$product_description %>% 
  word_tokenizer()

it = itoken(tokens, ids = data$id)
vocab = create_vocabulary(it)

#Now that we have a vocabulary, we can construct a document-term matrix
it = itoken(tokens, ids = data$id)
vectorizer = vocab_vectorizer(vocab)
dtm = create_dtm(it, vectorizer)

#Now we have a DTM and can check its dimensions.
str(dtm)

#Pruning vocabulary - remove both very common and very unusual terms
pruned_vocab = prune_vocabulary(vocab, term_count_min = 10, doc_proportion_max = 0.5, doc_proportion_min = 0.06)

it = itoken(tokens, ids = data$id)
vectorizer = vocab_vectorizer(pruned_vocab)
dtm = create_dtm(it, vectorizer)

#Check dimensions of reduced dtm
dim(dtm)

#Apply TF-IDF transformation to our DTM which will increase the weight of terms which are specific to
#a single document or handful of documents and decrease the weight for terms used in most documents
dtm = dtm %>% transform_tfidf()

#Find the Largest 100 Singular Values/Vectors of the dtm - reduced the number of features
k = 100
svds = svds(dtm, k=k)
m_truncated = dtm %*% svds$v

#Convert dtm to the reduced feature set
dtm = as.matrix(m_truncated)
colnames(dtm) = 1:k
colnames(dtm) = paste("description",colnames(dtm), sep="_")
data = cbind(data, dtm)

#Remove raw text columns
data[, ':=' (product_title = NULL,
             search_term = NULL,
             product_description = NULL)]

#Split data back into train and test
train_tf_idf = data[!is.na(relevance)]
test_tf_idf = data[is.na(relevance)]

#Save target vector
relevance_tf_idf = train_tf_idf[, relevance]

#Remove target vector from feature set
train_tf_idf[, relevance := NULL]
test_tf_idf[, relevance := NULL]

#Remove any zero variance columns
trainZeroVar = names(train_tf_idf[, sapply(train_tf_idf, function(v) var(v, na.rm=TRUE) == 0), with = F])
for (f in trainZeroVar) {
  train_tf_idf[, (f) := NULL]
  test_tf_idf[, (f) := NULL]
}

#Remove highly correlated features
trainCor = cor(train_tf_idf, use = "complete.obs")
trainCor[is.na(trainCor)] = 0
trainCor[upper.tri(trainCor)] = 0
diag(trainCor) = 0

trainRemove = names(train_tf_idf[,apply(trainCor,2,function(x) any(x >= 0.999 | x <= -0.999)), with = F])

train_tf_idf[, (trainRemove) := NULL]
test_tf_idf[, (trainRemove) := NULL]

#Save train and test as RData
save(list = c("train_tf_idf", "test_tf_idf", "relevance_tf_idf"), file = "feature_engineering_tf_idf.RData")

#Detach all packages from session to avoid conficts with other scripts
detachAllPackages()

#Remove all objects from the workspace
rm(list = ls())
gc()

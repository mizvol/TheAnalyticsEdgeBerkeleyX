ebayData = read.csv("eBayiPadTrain.csv", stringsAsFactors=FALSE)
ebayKaggle = read.csv("eBayiPadTest.csv", stringsAsFactors=FALSE)


#prepare corpus for description
library(tm)

corpus = Corpus(VectorSource(c(ebayData$description, ebayKaggle$description)))
corpus = tm_map(corpus, content_transformer(tolower), lazy=TRUE)
corpus = tm_map(corpus, PlainTextDocument, lazy=TRUE)
corpus = tm_map(corpus, removePunctuation, lazy=TRUE)
corpus = tm_map(corpus, removeWords, stopwords("english"), lazy=TRUE)
corpus = tm_map(corpus, stemDocument, lazy=TRUE)

#DTM
dtm = DocumentTermMatrix(corpus)

#Try different coefficient for sparse terms
sparse = removeSparseTerms(dtm, 0.98)
sparse
DescriptionWords = as.data.frame(as.matrix(sparse))
colnames(DescriptionWords) = make.names(colnames(DescriptionWords))

#split back to data and kaggle
DescriptionWordsData = head(DescriptionWords, nrow(ebayData))
DescriptionWordsKaggle = tail(DescriptionWords, nrow(ebayKaggle))

#add back original columns
DescriptionWordsData$sold = ebayData$sold
DescriptionWordsData$biddable = ebayData$biddable
DescriptionWordsData$startprice = ebayData$startprice
#DescriptionWordsData$cellular = ebayData$cellular
DescriptionWordsData$condition = ebayData$condition
#DescriptionWordsData$carrier = ebayData$carrier
#DescriptionWordsData$color = ebayData$color
#DescriptionWordsData$storage = NULL
#DescriptionWordsData$productline = NULL

#CART Dataset
DescriptionWordsKaggle$sold = ebayKaggle$sold
DescriptionWordsKaggle$biddable = ebayKaggle$biddable
DescriptionWordsKaggle$startprice = ebayKaggle$startprice
DescriptionWordsKaggle$condition = ebayKaggle$condition
DescriptionWordsKaggle$UniqueID = ebayKaggle$UniqueID

norm = (DescriptionWordsData$startprice-min(DescriptionWordsData$startprice))/(max(DescriptionWordsData$startprice)-min(DescriptionWordsData$startprice))
DescriptionWordsData$startprice = norm

#norm = (ebayData$startprice-min(ebayData$startprice))/(max(ebayData$startprice)-min(ebayData$startprice))

norm = (DescriptionWordsKaggle$startprice-min(DescriptionWordsKaggle$startprice))/(max(DescriptionWordsKaggle$startprice)-min(DescriptionWordsKaggle$startprice))
DescriptionWordsKaggle$startprice = norm

#norm = (ebayKaggle$startprice-min(ebayKaggle$startprice))/(max(ebayKaggle$startprice)-min(ebayKaggle$startprice))

#Split ebayData to train and test sets
set.seed(13)

spl = sample.split(DescriptionWordsData$sold, 0.7)

train = subset(DescriptionWordsData, spl == TRUE)


test = subset(DescriptionWordsData, spl == FALSE)


trainData = DescriptionWordsData
trainData = train
testData = test
testData = DescriptionWordsKaggle
#1. CART

library(rpart)
library(rpart.plot)

ebayCART = rpart(sold ~., data = trainData, method = "class", cp=0.0005)
#ebayCART = rpart(sold ~ biddable + startprice, data = trainData, method = "class", cp=0.000005)
prp(ebayCART)

#prediction
predCART = predict(ebayCART, newdata = testData)

#Compute AUC
predCART.prob = predCART[,2]
predROCR = prediction(predCART.prob, testData$sold)
#performance
perfROCR = performance(predROCR, "tpr", "fpr")
performance(predROCR, "auc")@y.values
probability = predCART.prob

#2. RF
library(randomForest)
trainData$sold = as.factor(trainData$sold)
trainData$condition = as.factor(trainData$condition)

testData$sold = as.factor(testData$sold)
testData$condition = as.factor(testData$condition)
ebayForest = randomForest(sold ~ ., data = trainData, ntree=1000)



predictForest = predict(ebayForest, type="prob", newdata = testData)
predForest.prob = predictForest[,2]
predROCR = prediction(predForest.prob, testData$sold)
perfROCR = performance(predROCR, "tpr", "fpr")
performance(predROCR, "auc")@y.values
probability = predictForest[,2]

#3. Logistic Regression

ebayReg = glm(sold ~., data = trainData, family = "binomial")
predReg = predict(ebayReg, newdata = testData, type = "response")
predROCR = prediction(predReg, testData$sold)
perfROCR = performance(predROCR, "tpr", "fpr")
performance(predROCR, "auc")@y.values

#Submission
MySubmission = data.frame(UniqueID = DescriptionWordsKaggle$UniqueID, Probability1 = probability)
MySubmission = data.frame(UniqueID = ebayKaggle$UniqueID, Probability1 = probability)

write.csv(MySubmission, "SubmissionSimpleLog.csv", row.names=FALSE)

library(party)

### Titanic: Machine Learning from Disaster ###
# Conditional inference trees solution
setwd("/your-path/titanic-ml")

# reading feature engineered data
train <- read.csv("./engineered-data/train_notenc.csv")
test <- read.csv("./engineered-data/test_notenc.csv")

# reading raw data
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")

train <- train[,!(names(train) %in% c('Survived'))]

# joining datasets for processing
combi <- rbind(train, test)

# factorizing variables
combi['deck'] <- factor(combi[['deck']])
combi['Sex'] <- factor(combi[['Sex']])
combi['room_pos'] <- factor(combi[['room_pos']])

# splitting data again
new_train <- combi[1:length(train[,1]),]
new_test <- combi[(length(train[,1])+1):length(combi[,1]),]

# removing unwanted features
drops <- c("PassengerId", "Cabin", "Name", "Ticket")
new_train <- new_train[,!(names(new_train) %in% drops)]
new_test <- new_test[,!(names(new_test) %in% drops)]

# appending dependent variable
new_train['Survived'] <- train_raw['Survived']

# training...
set.seed(0)
fit <- cforest(as.factor(Survived) ~ ., data = new_train, controls=cforest_unbiased(ntree=2000, mtry=3))

# predicting...
predictions <- predict(fit, new_test, OOB=TRUE, type = "response")

# preparing and writing submission...
submit <- data.frame(PassengerId = 892:1309, Survived = predictions)
write.csv(submit, 'sub27.csv', row.names=FALSE, quote=FALSE)

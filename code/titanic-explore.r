library(ggplot2)

setwd("~/shared/Projetos/Kaggle/titanic-ml")

train = read.csv("./engineered-data/train_notenc.csv")
test = read.csv("./engineered-data/test_notenc.csv")
torig = read.csv("train.csv")

# sex, age, fare and class
ggplot(aes(x=train$Sex, y=train$Age, color=as.factor(as.factor(train$Survived))), data=train) + geom_point()
ggplot(aes(x=train$Sex, y=train$Fare, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(x=train$Age, y=train$Fare, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(x=train$family_size, y=train$Fare, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(x=train$family_size, y=train$Age, color=as.factor(train$Survived)), data=train) + geom_point()

#position on the boat
ggplot(aes(x=train$Fare, y=train$room_pos, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(x=train$Age, y=train$room_pos, color=as.factor(train$Survived)), data=train) + geom_point()

ggplot(aes(y=train$Fare, x=train$deck, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(y=train$Age, x=train$deck, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(y=train$family_size, x=train$deck, color=as.factor(train$Survived)), data=train) + geom_point()

# title
ggplot(aes(x=train$Title, y=train$Age, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(x=train$Title, y=train$Fare, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(x=train$Title, y=train$family_size, color=as.factor(train$Survived)), data=train) + geom_point()

#families
ggplot(aes(x=train$Surname, y=train$Age, color=as.factor(train$Survived)), data=train) + geom_point()
ggplot(aes(x=train$Surname, y=train$Fare, color=as.factor(train$Survived)), data=train) + geom_point()


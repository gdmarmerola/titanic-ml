''' Kaggle challenge: predict survivors on the titanic disaster: Visualizations and analysis '''
from base import *
from preprocess import *

# set working directory
os.chdir("/your-path/titanic-ml")

# load data
train_df = pd.read_csv('train_notenc.csv')
test_df = pd.read_csv('test_notenc.csv')

from __future__ import division

# nulls and NAs
print "Train dataset nulls: \n"
for key in train_df.keys():
    print key, sum(train_df.loc[:, key].notnull())/len(train_df) * 100

print "Test dataset nulls: \n"
for key in test_df.keys():
    print key, sum(test_df.loc[:, key].notnull())/len(test_df) * 100

# compare test and train distributions
plt.figure(1)
plt.subplot(321)
train_df['Age'].hist(bins=40)
plt.subplot(322)
test_df['Age'].hist(bins=40)
plt.subplot(323)
train_df['Pclass'].hist(bins=3)
plt.subplot(324)
test_df['Pclass'].hist(bins=3)
plt.subplot(325)
train_df['Fare'].hist(bins=100)
plt.subplot(326)
test_df['Fare'].hist(bins=100)

# frequency tables
print train_df['Pclass'].value_counts()
print test_df['Pclass'].value_counts()

print train_df['deck'].value_counts()
print test_df['deck'].value_counts()

print train_df['room_pos'].value_counts()
print test_df['room_pos'].value_counts()

# sex, age, fare and class
ggplot(aes(x="Sex", y="Age", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="Sex", y="Fare", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="Age", y="Fare", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="family_size", y="Fare", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="family_size", y="Age", color="Survived"), data=train_df) + geom_point()

# position on the boat: front, middle, rear
ggplot(aes(x="Fare", y="room_pos", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="Age", y="room_pos", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="Pclass", y="room_pos", color="Survived"), data=train_df) + geom_point()

# position on the boat: deck number
ggplot(aes(x="Fare", y="deck", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="Age", y="deck", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="Pclass", y="deck", color="Survived"), data=train_df) + geom_point()
ggplot(aes(x="family_size", y="deck", color="Survived"), data=train_df) + geom_point()

# training features:
rmv = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']
dmatrix = train_df.drop(rmv, 1)
for i, element in enumerate(dmatrix.keys()):
    print i, element

# frequency tables:
feat = 'Surname'
train_df[feat].value_counts()

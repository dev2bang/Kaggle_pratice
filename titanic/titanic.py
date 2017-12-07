# from www.datacamp.com
# open course (kaggle-python-on-machine-learning)

import pandas as pd
import numpy as np
from sklearn import tree

# Load the train, test dataset from CSV (www.kaggle.com)
train_file = "train.csv"
test_file = "test.csv"
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# Cleaning and Formatting Data
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train["Age"] = train["Age"].fillna(train["Age"].median())
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
train["Age"] = train["Age"].fillna(train["Age"].median())
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# Select and create features for decision tree
target = train["Survived"].values
train_features = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "FamilySize"]].values

# Create and fit decision tree
my_tree_one = tree.DecisionTreeClassifier();
my_tree_one = my_tree_one.fit(train_features, target)

# Predict result using test DataSet
test.loc[152, "Fare"] = test["Fare"].median()
test["Age"] = test["Age"].fillna(test["Age"].median())
test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "FamilySize"]].values

my_prediction = my_tree_one.predict(test_features)
PassengerId = np.array(test["PassengerId"]).astype(int)
predict_result = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_tree_one.score(train_features, target))

predict_result.to_csv("result.csv", index_label = ["PassengerId"])

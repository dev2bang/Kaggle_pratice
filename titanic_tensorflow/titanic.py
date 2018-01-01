import tensorflow as tf
import pandas as pd
import numpy as np

# define number and data file name
LEARNING_COUNT = 1000
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
RESULT_FILE_NAME = "result.csv"

# loading data using pandas
train_data = pd.read_csv("train.csv")
test_data  = pd.read_csv("test.csv")

# for learning, convert to number
train_data.loc[train_data["Sex"]=="male", "Sex"] = 0
train_data.loc[train_data["Sex"]=="female", "Sex"] = 1
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1

test_data.loc[test_data["Sex"]=="male", "Sex"] = 0
test_data.loc[test_data["Sex"]=="female", "Sex"] = 1
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1


# For learning, do one hot encoding
x_data = train_data[["Pclass","Sex","Age","Fare","SibSp","Parch","FamilySize"]].values
test_x = test_data[["Pclass","Sex","Age","Fare","SibSp","Parch","FamilySize"]].values
y_data = pd.get_dummies(pd.Series(train_data["Survived"].values))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([7, 10], -1, 1.))
b1 = tf.Variable(tf.zeros([10]))
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

W2 = tf.Variable(tf.random_uniform([10, 20], -1, 1.))
b2 = tf.Variable(tf.zeros([20]))
L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)

W3 = tf.Variable(tf.random_uniform([20, 2], -1, 1.))
b3 = tf.Variable(tf.zeros([2]))
model = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range (1,LEARNING_COUNT):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})

prediction = tf.argmax(model,1)
target = tf.argmax(Y,1)

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("train data accuracy : %.2f" % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))

# Predict
result = sess.run(prediction, feed_dict={X: test_x})
PassengerId =np.array(test_data["PassengerId"]).astype(int)
my_solution = pd.DataFrame(result, PassengerId, columns = ["Survived"])
print(my_solution)

my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

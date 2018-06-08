"""
Iris Dataset
이번 실습은 붓꽃(iris, 아이리스) 데이터를 활용하여 붓꽃의 세부 종류를 구분하는 문제를 풉니다.

붓꽃의 종류는 크게 iris setosa / iris versicolor / iris virginica가 존재하며,
주어진 꽃잎과 꽃받침의 길이와 너비를 활용해 해당 꽃의 종류를 맞추는 알고리즘을
Single-layer Neural Network로 해결하면 됩니다.

각 컬럼에 대한 설명은 다음과 같습니다. 출처: ai-times

sepal length (cm): 꽃받침의 길이
sepal width (cm): 꽃받침의 너비
petal length (cm): 꽃잎의 길이
petal width (cm): 꽃잎의 너비
species: 붓꽃의 종류. iris setosa(0) / iris versicolor(1) / iris virginica(2) 의 세 종류가 있다.
"""
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris["data"]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"], random_state=0)

data = pd.DataFrame(X_train, columns=iris["feature_names"])
data["species"] = y_train

train_num_species = len(np.unique(y_train))
y_train_hot = np.eye(train_num_species)[y_train]

test_num_species = len(np.unique(y_test))
y_test_hot = np.eye(test_num_species)[y_test]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


num_epoch = 100000
learning_rate = 0.0015

w = np.random.uniform(low=-1.0, high=1.0, size=(4, 3))
b = np.random.uniform(low=-1.0, high=1.0, size=(1, 3))

for epoch in range(num_epoch):
    y_train_predict_hot = X_train.dot(w) + b
    y_train_predict_hot = sigmoid(y_train_predict_hot)

    y_train_predict = y_train_predict_hot.argmax(axis=1)
    accuracy = (y_train_predict == y_train).mean()

    # if epoch % 10 == 0:
    #     print(f"{epoch:2} accuracy = {accuracy:.5f}")

    w = w - learning_rate * X_train.T.dot(y_train_predict_hot - y_train_hot)
    b = b - learning_rate * (y_train_predict_hot - y_train_hot).mean(axis=0)

print(f"Train Data accuracy = {accuracy:.5f}")

y_test_predict_hot = X_test.dot(w) + b
y_test_predict_hot = sigmoid(y_test_predict_hot)

y_test_predict = y_test_predict_hot.argmax(axis=1)
accuracy = (y_test_predict == y_test).mean()

print("----" * 10)
print(f"Test Data accuracy = {accuracy:.5f}")
"""
result = pd.DataFrame(X, columns=iris["feature_names"])
result["species(actual)"] = y
result["species(predict)"] = y_predict
print(result.head(10))
"""
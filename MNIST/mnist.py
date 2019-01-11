"""
MNIST Dataset
이번 과제는 MNIST 필기체 데이터셋을 활용하여 필기체 이미지를 인식하는 이미지 인식 알고리즘을
Single-layer Neural Network로 작성합니다.

가로 28px, 세로 28px의 필기체 이미지가 주어지며, 필기체는 숫자 0부터 9까지 총 10개의 Label로 구성되어 있습니다.
이미지는 컬러가 없는 흑백 데이터이며, 한 픽셀의 값은 0 ~ 255 입니다. (0일수록 어둡고, 255일수록 밝습니다)

데이터는 60,000개의 Train 데이터와 10,000개의 Test 데이터가 주어지는데,
Train 데이터로 Single-layer Neural Network를 학습한 뒤 Test 데이터로 정확도(accuracy)를 측정합니다.

각 변수의 세부 정보는 다음과 같습니다.

X_train: Train 데이터의 Feature 입니다. 가로 28px, 세로 28px, 총 60,000개의 데이터로 구성되어 있습니다.
         픽셀 하나의 값은 0 ~ 255 입니다. (0일수록 어둡고, 255일수록 밝습니다)
y_train: Train 데이터의 Label 입니다. 총 60,000 개이며, 이미지가 어떤 숫자를 나타내는지가 적혀 있습니다.
         값은 0부터 9까지 입니다.
X_test: Test 데이터의 Feature 입니다. 가로 28px, 세로 28px, 총 10,000개의 데이터로 구성되어 있습니다.
        픽셀 하나의 값은 0 ~ 255 입니다. (0일수록 어둡고, 255일수록 밝습니다)
y_test: Test 데이터의 Label 입니다. 총 10,000 개이며, 이미지가 어떤 숫자를 나타내는지가 적혀 있습니다.
        값은 0부터 9까지 입니다.

주의 사항

이전에 Iris Dataset 문제를 풀었던 코드를 조금만 응용하면 매우 쉽게 MNIST 데이터셋 문제를 풀 수 있습니다.
accuracy가 잘 올라가지 않고 그 이유를 잘 모르겠다면, Loss Function(=Cross Entropy)를 병행해서 사용해보세요.
앞서 언급드린대로 Loss Function은 "학습이 잘 될수록 0에 수렵하고, 학습이 잘 되지 않을수록 무한대로 발산합니다."
즉, Loss Function을 사용할 결과가 무한대로 발산하고 있다면 무언가 제대로 풀리지 않고 있다는 것입니다.
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import to_categorical

((X_train, y_train), (X_test, y_test)) = mnist.load_data()
"""
plt.gray()

print(y_train[0:10])

figures, axes = plt.subplots(nrows=2, ncols=5)
figures.set_size_inches(18, 8)

axes[0][0].matshow(X_train[0])
axes[0][1].matshow(X_train[1])
axes[0][2].matshow(X_train[2])
axes[0][3].matshow(X_train[3])
axes[0][4].matshow(X_train[4])
axes[1][0].matshow(X_train[5])
axes[1][1].matshow(X_train[6])
axes[1][2].matshow(X_train[7])
axes[1][3].matshow(X_train[8])
axes[1][4].matshow(X_train[9])

plt.show()
"""
X_train = X_train.reshape(60000, 28 * 28)
X_test = X_test.reshape(10000, 28 * 28)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(actual, predict, eps=1e-15):
    actual = np.array(actual)
    predict = np.array(predict)

    clipped_predict = np.minimum(np.maximum(predict, eps), 1 - eps)

    loss = actual * np.log(clipped_predict) + (1 - actual) * np.log(1 - clipped_predict)

    return -1.0 * loss.mean()


num_epoch = 100000
learning_rate = 0.00000003

w = np.random.uniform(low=-1.0, high=1.0, size=(28 * 28, 10))
b = np.random.uniform(low=-1.0, high=1.0, size=(1, 10))

# print(y_train[0:10])

for epoch in range(num_epoch):
    y_train_predict_hot = X_train.dot(w) + b
    y_train_predict_hot = sigmoid(y_train_predict_hot)

    y_train_predict = y_train_predict_hot.argmax(axis=1)
    cross_ntropy = cross_entropy(y_train_hot, y_train_predict_hot)
    accuracy = (y_train_predict == y_train).mean()

    if epoch % 10 == 0:
        # print(y_train_predict[0:10])
        print(f"{epoch:2} accuracy = {accuracy:.5f} cross_entropy = {cross_ntropy:.5f}")

    w = w - learning_rate * X_train.T.dot(y_train_predict_hot - y_train_hot)
    b = b - learning_rate * (y_train_predict_hot - y_train_hot).mean(axis=0)

y_test_predict_hot = X_test.dot(w) + b
y_test_predict_hot = sigmoid(y_test_predict_hot)

y_test_predict = y_test_predict_hot.argmax(axis=1)
accuracy = (y_test_predict == y_test).mean()

print(f"Test accuracy = {accuracy:.5f}")

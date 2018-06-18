# Conv2D
# MaxPooling2D
# Flatten
# Dense
# SGD
import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.utils import to_categorical


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(actual, predict, eps=1e-15):
    actual = np.array(actual)
    predict = np.array(predict)

    clipped_predict = np.minimum(np.maximum(predict, eps), 1 - eps)

    loss = actual * np.log(clipped_predict) + (1 - actual) * np.log(1 - clipped_predict)

    return -1.0 * loss.mean()


def conv2d(X_train, k=2, f=3, s=1, p=1):
    # num of filter
    # size of filter
    # stride
    # zero padding
    X = np.zeros((X_train.shape[0], X_train.shape[1]+(p*2), X_train.shape[2]+(p*2)))
    for i in range(int(X_train.shape[0])):
        X[i] = np.pad(X_train[i], pad_width=1, mode='constant', constant_values=0)

    w1 = np.random.uniform(low=-1.0, high=1.0, size=(f, f))
    b1 = np.random.uniform(low=-1.0, high=1.0, size=(1, 1))

    size = int(((X_train.shape[1] - f + (2 * p)) / s) + 1)
    z1 = np.zeros((X_train.shape[0], size, size))
    conv1 = np.zeros((size, size))
    for num in range(X_train.shape[0]):
        for i in range(size):
            x_stride = s*i
            for j in range(size):
                y_stride = s*j
                conv1[i][j] = np.sum(X[num, x_stride:f+x_stride, y_stride:f+y_stride] * w1) + b1
        a1 = sigmoid(conv1)
        if num % 100 == 0:
            print(f"conv epoch : {num}")
        z1[num] = a1

    return z1


def max_pooling(X, f=2, s=2):
    size = int(((X.shape[1] - f) / s) + 1)
    z1 = np.zeros((X.shape[0], size, size))
    conv1 = np.zeros((size, size))
    print(z1.shape)
    print(conv1.shape)
    for num in range(X.shape[0]):
        for i in range(size):
            x_stride = s * i
            for j in range(size):
                y_stride = s * j
                conv1[i][j] = np.max(X[num, x_stride:f+x_stride, y_stride:f+y_stride])
        z1[num] = conv1
        if num % 100 == 0:
            print(f"max_pooling epoch : {num}")

    return z1


def gradient_descent(X_train, y, y_hot, num_epoch=10, learning_rate=1.0):
    num_features = X_train.shape[1]
    num_hidden_layer = 1000

    # w = np.random.uniform(low=-1.0, high=1.0, size=num_features)
    # b = np.random.uniform(low=-1.0, high=1.0)

    w1 = np.random.uniform(low=-1.0, high=1.0, size=(num_features, num_hidden_layer))
    b1 = np.random.uniform(low=-1.0, high=1.0, size=(1, num_hidden_layer))

    w2 = np.random.uniform(low=-1.0, high=1.0, size=(num_hidden_layer, 10))
    b2 = np.random.uniform(low=-1.0, high=1.0, size=(1, 10))

    for epoch in range(num_epoch):
        # forward propagation
        z1 = X_train.dot(w1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(w2) + b2
        a2 = sigmoid(z2)

        predict = np.argmax(a2, axis=1)
        cross_ntropy = cross_entropy(y_hot, a2)
        accuracy = (predict == y).mean()

        print(f"{epoch:2} accuracy = {accuracy:.6f} cross_entropy = {cross_ntropy:.6f}")

        if accuracy == 1.0:
            break

        # back propagation
        d2 = a2 - y_hot
        d1 = d2.dot(w2.T) * a1 * (1 - a1)

        w2 = w2 - learning_rate * a1.T.dot(d2)
        w1 = w1 - learning_rate * X_train.T.dot(d1)
        b2 = b2 - learning_rate * d2.mean(axis=0)
        b1 = b1 - learning_rate * d1.mean(axis=0)

    return accuracy, cross_ntropy


def main():
    ((X_train, y_train), (X_test, y_test)) = mnist.load_data()

    # X_train = X_train.reshape(60000, 28 * 28)
    # X_test = X_test.reshape(10000, 28 * 28)
    # y_train_hot = to_categorical(y_train)
    # y_test_hot = to_categorical(y_test)
    # print("----" * 10)
    # print(f"accuracy = {accuracy:.6f} cross_entropy = {cross_ntropy:.6f}")
    # print(X_train.shape)
    # print(X_train)
    # print(y_train_hot.shape)
    # print(y_train_hot)
    # print(X_train)
    X_train_tmp = conv2d(X_train)
    X_train_tmp = conv2d(X_train_tmp)
    X_train_tmp = max_pooling(X_train_tmp)

    X_train_tmp = conv2d(X_train_tmp)
    X_train_tmp = conv2d(X_train_tmp)
    X_train_tmp = max_pooling(X_train_tmp)
    X_train_tmp.shape

    X_train_tmp = X_train_tmp.reshape(60000, X_train_tmp.shape[1] * X_train_tmp.shape[2])
    X_test = X_test.reshape(10000, 28 * 28)
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    accuracy, cross_ntropy = gradient_descent(X_train_tmp, y_train, y_train_hot)
    print("----" * 10)
    print(f"accuracy = {accuracy:.6f} cross_entropy = {cross_ntropy:.6f}")


if __name__ == '__main__':
    main()

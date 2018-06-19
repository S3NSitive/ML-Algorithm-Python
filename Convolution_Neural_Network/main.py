import numpy as np

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


def my_conv(X_train, k=2, f=3, s=1, p=1):
    X = np.zeros((X_train.shape[0], X_train.shape[1]+(p*2), X_train.shape[2]+(p*2)))
    for i in range(int(X_train.shape[0])):
        X[i] = np.pad(X_train[i], pad_width=1, mode='constant', constant_values=0)

    X = X.reshape(60000, X.shape[1], X.shape[2], 1)

    w = np.random.uniform(low=-1.0, high=1.0, size=(f, f, 1))
    b = np.random.uniform(low=-1.0, high=1.0, size=(1, 1, 1))

    size = int(((X_train.shape[1] - f + (2 * p)) / s) + 1)
    z1 = np.zeros((X_train.shape[0], size, size))
    d = np.zeros((X_train.shape[0], size, size))

    for _ in range(2):
        for num in range(X_train.shape[0]):
            for i in range(size):
                x = s * i
                for j in range(size):
                    y = s * j
                    d[i][j] = np.sum(X[num, x:f + x, y:f + y, :] * w) + b
            a1 = sigmoid(d)
            if num % 100 == 0:
                print(f"conv epoch : {num}")
            z1[num] = a1

        max = max_pooling(z1)
        X, w = back_prop(z1, max, w)


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


def back_prop(X, max, w):
    X_back = X.copy()
    new_w = w.copy()
    new_x = X.copy()
    f = 2
    for num in range(X_back.shape[0]):
        for i in range(max.shape[1]):
            x = 2 * i
            for j in range(max.shape[2]):
                y = 2 * j
                arridx1 = np.where(X_back[num, x:f + x, y:f + y] == max[num, i, j])
                if (i == arridx1[0] and j == arridx1[1]):
                    continue
                else:
                    X_back[i][j] = 0

    w180 = np.rot90(w, 2)
    pad_bd = np.pad(X_back, pad_width=1, mode='constant', constant_values=0)

    for i in range(2):
        for j in range(2):
            new_w[i][j] = np.sum(X_back[0, i:i + 2, j:j + 2] * X[0, i:i + 2, j:j + 2])

    for num in range(X_back.shape[0]):
        for i in range(3):
            for j in range(3):
                new_x[0][i][j] = np.sum(pad_bd[i:i + 2, j:j + 2] * w180)

    return new_w, new_x


def main():
    ((X_train, y_train), (X_test, y_test)) = mnist.load_data()

    my_conv(X_train)


if __name__ == '__main__':
    main()

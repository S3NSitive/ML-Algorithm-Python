import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

# relu_back
# maxpooling_back
# fc_for, fc_back

def relu_forward(x):
    return x * (x > 0)


def relu_backward(x):
    return


def cross_entropy(actual, predict, eps=1e-15):
    actual = np.array(actual)
    predict = np.array(predict)

    clipped_predict = np.minimum(np.maximum(predict, eps), 1 - eps)
    loss = actual * np.log(clipped_predict) + (1 - actual) * np.log(1 - clipped_predict)

    return -1.0 * loss.mean()


def flatten_forward(x):
    size = x.shape[0]

    return x.reshape(size, -1)


def flatten_backward(x):
    size = x.shape[0]
    pass


def conv_forward(X, w, b, s=1, p=1):
    N, H, W, C = X.shape
    F, wH, wW, _ = w.shape

    pad_size = p*2
    X_pad = np.zeros((N, H+pad_size, W+pad_size, C))
    for _n in range(N):
        for _c in range(C):
            X_pad[_n, :, :, _c] = np.pad(X[_n, :, :, _c], pad_width=1, mode='constant', constant_values=0)

    H_size = 1 + (H + pad_size - wH) // s
    W_size = 1 + (W + pad_size - wW) // s
    result = np.zeros((N, H_size, W_size, F))

    for _n in range(N):
        for _h in range(H_size):
            for _w in range(W_size):
                for _f in range(F):
                    result[_n, _h, _w, _f] = np.sum(X_pad[_n, _h*s:_h*s+wH, _w*s:_w*s+wW, :] * w[_f]) + b[_f]
        if _n % 100 == 0:
            print(f"conv epoch : {_n}")

    return result


def conv_backward(X_out, X, w, b, s=1, p=1):
    N_out, H_out, W_out, C_out = X_out.shape
    N, H, W, C = X.shape
    F, wH, wW, wC = w.shape

    pad_size = p * 2
    X_out_pad = np.zeros((N_out, H_out + pad_size, W_out + pad_size, C_out))
    for _n in range(N_out):
        for _c in range(C_out):
            X_out_pad[_n, :, :, _c] = np.pad(X_out[_n, :, :, _c], pad_width=1, mode='constant', constant_values=0)

    X_pad = np.zeros((N, H + pad_size, W + pad_size, C))
    for _n in range(N):
        for _c in range(C):
            X_pad[_n, :, :, _c] = np.pad(X[_n, :, :, _c], pad_width=1, mode='constant', constant_values=0)

    w180 = np.rot90(w, 2)

    H_size = 1 + (H + pad_size - wH) // s
    W_size = 1 + (W + pad_size - wW) // s
    new_w = np.zeros(w.shape)
    new_x = np.zeros((N, H, W, C))
    new_b = np.zeros((F))

    for _n in range(N):
        for _h in range(H_out):
            for _w in range(W_out):
                new_b = new_b + X_out[_n, _h, _w, :]

    for _n in range(N):
        for _h in range(wH):
            for _w in range(wW):
                for _f in range(F):
                    new_w[_n, _h, _w, _f] = np.sum(X_pad[_n, _h*s:_h*s+H, _w*s:_w*s+W, :] * X_out[_n, _h*s:_h*s+H, _w*s:_w*s+W, :])

    for _n in range(N):
        for _h in range(H_size):
            for _w in range(W_size):
                for _f in range(F):
                    new_x[_n, _h, _w, _f] = np.sum(X_out_pad[_n, _h*s:_h*s+wH, _w*s:_w*s+wW, :] * w180[_f]) + b[_f]

    return new_w, new_x, new_b


def max_pooling_forward(X, f=2, s=2):
    N, H, W, C = X.shape

    H_size = 1 + (H - f) // s
    W_size = 1 + (W - f) // s
    result = np.zeros((N, H_size, W_size, C))

    for _n in range(N):
        for _h in range(H_size):
            for _w in range(W_size):
                for _c in range(C):
                    result[_n, _h, _w, _c] = np.max(X[_n, _h*s:_h*s+f, _w*s:_w*s+f, _c])
        if _n % 100 == 0:
            print(f"max_pooling epoch : {_n}")

    return result


# def max_pooling_backward(X, f=2, s=2):
#     for num in range(X_back.shape[0]):
#         for i in range(max.shape[1]):
#             x = 2 * i
#             for j in range(max.shape[2]):
#                 y = 2 * j
#                 arridx1 = np.where(X_back[num, x:f + x, y:f + y] == max[num, i, j])
#                 if (i == arridx1[0] and j == arridx1[1]):
#                     continue
#                 else:
#                     X_back[i][j] = 0


def main():
    ((X_train, y_train), (X_test, y_test)) = mnist.load_data()

    X_train = X_train.reshape(60000, 28, 28, 1)
    # X_test = X_test.reshape(10000, 28 * 28)
    # y_train_hot = to_categorical(y_train)
    # y_test_hot = to_categorical(y_test)

    num_epoch = 100
    learning_rate = 1.0

    w1 = np.random.normal(0, np.sqrt(2 / 1*3*3), (4, 3, 3, 1))
    b1 = np.zeros(4)
    w2 = np.random.normal(0, np.sqrt(2 / 4*3*3), (8, 3, 3, 4))
    b2 = np.zeros(8)
    w3 = np.random.normal(0, np.sqrt(2 / 512), (512, 128))
    b3 = np.zeros(128)
    w4 = np.random.normal(0, np.sqrt(2 / 512), (128, 10))
    b4 = np.zeros(10)

    for epoch in range(num_epoch):
        z1 = conv_forward(X_train, w1, b1)
        a1 = relu_forward(z1)
        z2 = conv_forward(a1, w2, b2)
        a2 = relu_forward(z2)




if __name__ == '__main__':
    main()

from mxnet import nd
import random
from mxnet import autograd

true_w = [0.5, 1]
true_b = 0.8
num_features = 1000
num_inputs = 2
X = nd.random.normal(scale=0.01, shape=(num_features, num_inputs))
Y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b

#小批量读取
def data_iter(X, Y, batch_size):
    indices = list(range(num_features))
    random.shuffle(indices)
    for i in range(0, len(X), batch_size):
        j = nd.array(indices[i: min(i+batch_size, len(X))])
        yield X.take(j), Y.take(j)


#定义损失函数
def square_loss(y, y_hat):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2)/2

w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))

b = nd.zeros(shape=(1, ))
print(w, b)
w.attach_grad()
b.attach_grad()

#定义算法
def net(X, w, b):
    return nd.dot(X, w) + b

#定义优化算法
def sgd(lr, batch_size, params):
    for param in params:
        param[:] = param - lr * param.grad/batch_size

#开始训练
batch_size = 10
lr = 0.03
epochs = 20
for epoch in range(epochs):
    for feature, label in data_iter(X, Y, batch_size):
        with autograd.record():
            y_hat = net(feature, w, b)
            l = square_loss(label, y_hat)
        l.backward()
        sgd(lr, batch_size, params=[w, b])
    train_loss = square_loss(Y, net(X, w, b))
    print('epoch %d, loss %.3f' % (epoch + 1, train_loss.mean().asnumpy()))
print(w, b)



from mxnet import nd, autograd
import random


#手工数据
true_w = [1, 2]
true_b = 5
num_inputs = 2
num_examples = 1000
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

#小批量读取数据
def data_iter(batch_size, features, labels):
    len_input = len(features)
    indices = list(range(len_input))
    random.shuffle(indices)
    for i in range(0, len_input, batch_size):
        j = nd.array(indices[i:min(i+batch_size, len_input)])
        yield features.take(j), labels.take(j)

#定义损失函数
def squre_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

#定义sgd函数
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr*param.grad/batch_size

#定义模型
def net(X, w, b):
    return nd.dot(X, w) + b

#初始化参数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(1, )
w.attach_grad()
b.attach_grad()

#开始训练
lr = 0.03
batch_size = 256
epochs = 30
for epoch in range(epochs):
    for feature, label in data_iter(batch_size, features, labels):
        with autograd.record():
            y_hat = net(feature, w, b)
            l = squre_loss(y_hat, label)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_loss = squre_loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch+1, train_loss.mean().asnumpy()))
print(w, b)
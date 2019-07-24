from mxnet.gluon import data as gdata
from mxnet import nd, autograd
import d2lzh as d2l

#获取数据
train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)
transformer = gdata.vision.transforms.ToTensor()
batch_size = 256
train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size=batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size=batch_size)


#定义softmax
def softmax(x):
    x_exp = x.exp()
    return x_exp/x_exp.sum(axis=1, keepdims=True)


#定义模型
def net(x, w1, w2, b1, b2):
    H = nd.dot(x.reshape((-1, num_inputs)), w1) + b1
    H = relu(H)
    return softmax(nd.dot(H, w2) + b2)


def relu(x):
    return nd.maximum(x, 0)

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def loss(y_hat, y):
    return -nd.pick(y_hat, y).log()


def test_loss(data, net, w1, w2, b1, b2):
    test_l, n = 0.0, 0
    for x, y in data:
        y_hat = net(x, w1, w2, b1, b2)
        l = loss(y_hat, y).sum()
        test_l += l.asscalar()
        n += y.size
    return test_l/n

#定义并初始化参数
num_inputs = 784
num_hiddens = 256
num_outputs = 10
w1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
w2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
epochs = 10
lr = 0.1
w1.attach_grad()
w2.attach_grad()
b1.attach_grad()
b2.attach_grad()
train_plot, test_plot = [], []
for epoch in range(epochs):
    train_loss, n = 0.0, 0

    for feature, label in train_iter:
        with autograd.record():
            y_hat = net(feature, w1, w2, b1, b2)
            l = loss(y_hat, label).sum()
        l.backward()
        sgd([w1, w2, b1, b2], lr, batch_size)
        train_loss += l.asscalar()
        n += label.size
    train_plot.append(train_loss/n)
    test_loss1 = test_loss(test_iter, net, w1, w2, b1, b2)
    test_plot.append(test_loss1)
    print('epoch:%d, train_loss:%.3f, test_loss:%.3f' % (epoch + 1, train_loss/n, test_loss1))


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
        d2l.plt.show()

semilogy(range(1, epochs + 1), train_plot, 'epochs', 'loss',
         range(1, epochs + 1), test_plot, legend=['train', 'test'])




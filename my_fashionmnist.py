from mxnet.gluon import data as gdata
import d2lzh as d2l
from mxnet.gluon import nn
import matplotlib
from mxnet import init, autograd
from mxnet import gluon


train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)
batch_size = 256
data_iter = gdata.DataLoader(train_data, batch_size=batch_size, shuffle=True)
d2l.use_svg_display()
images, labels = train_data[0:9]


def get_fashionmnist_label(labels):
    texts = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [texts[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, image, label in zip(figs, images, get_fashionmnist_label(labels)):
        f.imshow(image.reshape((28, 28)).asnumpy())
        f.set_title(label)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    matplotlib.pyplot.show()


#读取数据
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size, shuffle=False)
#建立模型
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
#定义损失函数
loss = gluon.loss.SoftmaxCrossEntropyLoss()
#定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.3})


#定义精度
def acc(data, net):
    acc, n = 0.0, 0
    for x, y in data:
        y = y.astype('float32')
        acc += (net(x).argmax(axis=1) == y).sum().asscalar()
        n += y.size()
    return acc/n


epochs = 10
for epoch in range(epochs):
    train_loss, train_acc = 0.0, 0.0
    n = 0
    for feature, label in train_iter:
        with autograd.record():
            y_hat = net(feature)
            l = loss(y_hat, label).sum()
        label = label.astype('float32')
        l.backward()
        trainer.step(batch_size)
        train_loss += l.asscalar()
        train_acc += (y_hat.argmax(axis=1) == label).sum().asscalar()
        n += label.size
    test_acc = acc(test_iter, net)
    print('epoch %d, train_acc:%.3f, test_acc:%.3f' % (epoch + 1, train     _acc/n, test_acc))

for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
matplotlib.pyplot.show()



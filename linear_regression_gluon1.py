from mxnet import gluon, nd, autograd, init
import random


#生成数据
true_w = [0.2, 3]
true_b = 2
num_inputs = 2
num_examples = 1000
X = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
Y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b


net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
batch_size = 10
lr = 0.03
epochs = 5
loss = gluon.loss.L2Loss()
dataset = gluon.data.ArrayDataset(X, Y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
for epoch in range(epochs):
    for feature, label in data_iter:
        with autograd.record():
            l = loss(net(feature), label)
        l.backward()
        gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03}).step(batch_size)
    train_loss = loss(net(X), Y)
    print('epoch %d, train_loss: %.3f' % (epoch + 1, train_loss.mean().asnumpy()))

print(true_w, net[0].weight.data())
print(true_b, net[0].bias.data())

#add something to try how git works

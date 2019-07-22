from mxnet.gluon import data as gdata
import d2lzh as d2l

train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)
batch_size = 10
data_iter = gdata.DataLoader(train_data, batch_size=batch_size, shuffle=True)
d2l.use_svg_display()
images, labels = train_data[0:9]
_, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
for f, image, label in zip(figs, images, labels):
    f.imshow(image.reshape((28, 28)).asnumpy())

import paddle
from paddle.framework import dtype 
import paddle.nn as nn
import paddle.vision as vision
from paddle.static import InputSpec
import numpy as np
import os
import matplotlib.pyplot as plt



class MyLeNet(nn.Layer):

    def __init__(self, num_classes=10):
        super(MyLeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2D(
                1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(
                6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2))

        
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
            nn.Softmax())

    # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html
    # @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 1, 28, 28], dtype='float32')])
    def forward(self, inputs):
        x = self.features(inputs)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x


def one_hot_coding(labels):
    t = paddle.zeros((len(labels), 10), dtype='int64')
    for i, item in enumerate(labels):
        t[i, item] = 1

    return t


class DataSet(paddle.io.Dataset):
    def __init__(self, mode='train'):
        super(DataSet, self).__init__()
        self.all_data = np.load('./data/mnist.npz')
        self.mode = mode
        self.train_x = paddle.to_tensor(self.all_data['x_train']/255., dtype='float32').reshape((-1,1,28,28))
        self.test_x = paddle.to_tensor(self.all_data['x_test']/255., dtype='float32').reshape((-1,1,28,28))
        # self.train_y = one_hot_coding(self.all_data['y_train'])
        # self.test_y = one_hot_coding(self.all_data['y_test'])
        self.train_y = paddle.to_tensor(self.all_data['y_train'], dtype='int64')
        self.test_y = paddle.to_tensor(self.all_data['y_test'], dtype='int64')
        
    def __getitem__(self, idx):
        if self.mode == 'train':
            img = self.train_x[idx]
            label = self.train_y[idx]
            
        else:
            img = self.test_x[idx]
            label = self.train_y[idx]
            
        return img, label
    
    def __len__(self):
        if self.mode == 'train':
            # print(self.train_y.shape[0])
            return self.train_y.shape[0]
        else:
            # print(self.test_y.shape[0])
            return self.test_y.shape[0]


def train(layer, loader, loss_fn, opt, epoch):
    layer.train()
    for epoch_id in range(epoch):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))


# net = vision.models.LeNet()
net = MyLeNet()
paddle.summary(net, (1,1,28,28))
assert 1==2
# print(type(net))
# # print(type(paddle.Model(model)))
# # x = paddle.rand([1, 1, 28, 28])
# data = np.load('./data/mnist.npz')
# print(data['y_train'].shape[0])
# print(data['x_train'][0][14])
# idx = int(input("input idx: "))
# x = vision.transforms.to_tensor(data['x_test'][idx])
# x = x.reshape((1,1,28,28))
# out = net(x)


# parms = net.parameters()
# print(parms)
# print(paddle.nn.Softmax(out))

# print(out.shape)
# print(out)
# plt.imshow(np.squeeze(x))
# plt.title(str(data['y_test'][idx]))
# plt.show()
# assert 1==2

dynamic = True
if not dynamic:
    paddle.enable_static()
input = InputSpec([None, 1, 28, 28], 'float32', 'image')
label = InputSpec([None, 1], 'int64', 'label')

model = paddle.Model(net, input, label) # 静态图的话不能缺少input参数; 计算损失用到label的话不能缺少label参数
optim = paddle.optimizer.Adam(learning_rate=1e-3,
                            parameters=model.parameters())
model.prepare(optim,
             paddle.nn.CrossEntropyLoss(),
             paddle.metric.Accuracy())
transform = vision.transforms.Compose([
    vision.transforms.Transpose()
    #vision.transforms.Normalize(mean=[127.5, 127.5, 127.5])
])

train_data_gen = paddle.io.DataLoader(DataSet(mode='train'), batch_size=64, shuffle=True, drop_last=True, num_workers=0)
# model.fit(train_data_gen, epochs=2, verbose=1)
train(net, train_data_gen, paddle.nn.CrossEntropyLoss(), optim, 2)

test_data_gen = paddle.io.DataLoader(DataSet(mode='test'), batch_size=64, shuffle=True, drop_last=True)
model.evaluate(test_data_gen, verbose=1) # 在输入数据上，评估模型的损失函数值和评估指标

# model.save('checkpoint/test')  # save for training
# model.save('models/inference_model', False)  # save for inference

# 保存模型&参数
paddle.jit.save(net, 'models/dyn_model', input_spec=[InputSpec(shape=[None, 1, 28, 28], dtype='float32')])



# import paddle
# import paddle.vision.transforms as T
# from paddle.vision.datasets import MNIST
# from paddle.static import InputSpec

# dynamic = True
# if not dynamic:
#     paddle.enable_static()

# transform = T.Compose([
#       T.Transpose(),
#       T.Normalize([127.5], [127.5])
#   ])
# train_dataset = MNIST(mode='train', transform=transform)
# train_loader = paddle.io.DataLoader(train_dataset,
#     batch_size=64)
# val_dataset = MNIST(mode='test', transform=transform)
# val_loader = paddle.io.DataLoader(val_dataset,
#     batch_size=64)

# input = InputSpec([None, 1, 28, 28], 'float32', 'image')
# label = InputSpec([None, 1], 'int64', 'label')

# model = paddle.Model(
#     paddle.vision.models.LeNet(), input, label)
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=model.parameters())
# model.prepare(
#     optim,
#     paddle.nn.CrossEntropyLoss(),
#     paddle.metric.Accuracy(topk=(1, 2)))
# model.fit(train_loader,
#           val_loader,
#           epochs=10,
#           save_dir='mnist_checkpoint',
#           verbose=2)

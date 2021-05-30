import paddle
from paddle.framework import dtype 
import paddle.nn as nn
import paddle.vision as vision
import numpy as np
import os
import matplotlib.pyplot as plt



class LeNet(nn.Layer):

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
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

        if num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(400, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_classes))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        return x

# net = vision.models.LeNet()
net = LeNet()
paddle.summary(net, (1,1,28,28))
# print(type(net))
# # print(type(paddle.Model(model)))
# # x = paddle.rand([1, 1, 28, 28])
data = np.load('./data/mnist.npz')
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
# plt.show()


def one_hot_coding(labels):
    t = paddle.zeros((len(labels), 10), dtype='int64')
    for i, item in enumerate(labels):
        t[i, item] = 1

    return t

# print(one_hot_coding(data['y_train']))
# assert 1==0

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
            print(self.train_y.shape[0])
            return self.train_y.shape[0]
        else:
            print(self.test_y.shape[0])
            return self.test_y.shape[0]

model = paddle.Model(net) # 静态图的话不能缺少input参数
optim = paddle.optimizer.SGD(learning_rate=1e-3,
                            parameters=model.parameters())
model.prepare(optim,
             paddle.nn.CrossEntropyLoss(),
             paddle.metric.Accuracy())
transform = vision.transforms.Compose([
    vision.transforms.Transpose()
    #vision.transforms.Normalize(mean=[127.5, 127.5, 127.5])
])

train_data_gen = paddle.io.DataLoader(DataSet(mode='train'), batch_size=64, shuffle=True, drop_last=True, num_workers=0)
model.fit(train_data_gen, epochs=1, verbose=1)

test_data_gen = paddle.io.DataLoader(DataSet(mode='test'), batch_size=64, shuffle=True, drop_last=True)
model.evaluate(test_data_gen, verbose=1)
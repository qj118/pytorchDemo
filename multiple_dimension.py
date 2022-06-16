import numpy as np
import torch
import matplotlib.pyplot as plt

# 从文件读取数据
xy = np.loadtxt('./diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 取出每一行除最后一列的元素
y_data = torch.from_numpy(xy[:, [-1]])  # 取出每一行最后一个元素


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.ReLU()  # 中间激活函数使用 RelU
        self.sigmoid = torch.nn.Sigmoid()  # 最后一层二元分类：激活函数使用 sigmoid

    def forward(self, x):
        x = self.activate(self.linear1(x))  # 上一步是下一步的输入
        x = self.activate(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 构建模型，损失函数以及优化函数
model = Model()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []

for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


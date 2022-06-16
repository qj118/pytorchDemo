import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 继承 Module 类
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入特征的维度，输出特征的维度

    def forward(self, x):  # 一定要有的函数，model 是一个可调用类，调用时默认调用 forward 函数
        y_pred = self.linear(x)
        return y_pred


# 创建对象
model = LinearModel()
# 构造 Loss Function
criterion = torch.nn.MSELoss(size_average=False)  # 不求和
# 构造优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 将训练的参数和学习率传入

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)  # 返回一个 Tensor 对象
    print(epoch, loss.item())

    optimizer.zero_grad()  # backward 会累加梯度，所以需要每轮进行清零
    loss.backward()  # backward propagation
    optimizer.step()  # 更新

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# 预测
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

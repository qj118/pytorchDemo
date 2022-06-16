import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # 权重初始化为一个张量
w.requires_grad = True


# forward propagation
def forward(x):
    return x * w  # 当两者相乘 x 会自动类型转换成一个张量，而返回的自然也是一个张量


# loss function
def loss(x, y):
    return (forward(x) - y) ** 2


print('predict (before training)', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()  # 生成计算图，并自动回溯
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data  # 学习率为 0.01 更新权重，grad 也是一个张量
        w.grad.data.zero_()
    print('progress:', epoch, l.item())

print('predict (after training):', 4, forward(4).item())


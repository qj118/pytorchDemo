import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_1 = torch.tensor([1.0], requires_grad=True)  # 权重初始化为一个张量
w_2 = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)


# forward propagation
def forward(x):
    return x ** 2 * w_1 + x * w_2 + b


# loss function
def loss(x, y):
    return (forward(x) - y) ** 2


print('predict (before training)', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()  # 生成计算图，并自动回溯
        lr = 0.01
        print('\t grad:', x, y, w_1.grad.item(), w_2.grad.item(), b.grad.item())
        w_1.data = w_1.data - lr * w_1.grad.data
        w_2.data = w_2.data - lr * w_2.grad.data
        b.data = b.data - lr * b.grad.data

        w_1.grad.data.zero_()
        w_2.grad.data.zero_()
        b.grad.data.zero_()

    print('progress:', epoch, l.item())

print('predict (after training):', 4, forward(4).item())


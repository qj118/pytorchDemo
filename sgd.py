x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


# hypothesis
def forward(x):
    return w * x


# cost function
def loss(x, y):
    return (forward(x) - y) ** 2


# gradient descent
def gradient(x, y):
    return 2 * x * (forward(x) - y)


epoch_list = []
cost_list = []

# 预测结果
print('Predict (before training)', 4, forward(4))
for epoch in range(100):  # 100 轮迭代
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print('\tgrad:', x, y, grad, l)
    print('progess:', epoch, 'w = ', w)
print('Predict (after training)', 4, forward(4))



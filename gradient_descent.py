import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


# hypothesis
def forward(x):
    return w * x


# cost function
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# gradient descent
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (forward(x) - y)
    return grad / len(xs)


epoch_list = []
cost_list = []

# 预测结果
print('Predict (before training)', 4, forward(4))
for epoch in range(100):  # 100 轮迭代
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    epoch_list.append(epoch)
    cost_list.append(cost_val)
    print("Epoch:", epoch, 'w = ', w, 'loss = ', cost_val)
print('Predict (after training)', 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()


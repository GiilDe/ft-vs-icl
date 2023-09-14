import torch

m1 = torch.nn.Linear(1, 1)

m1.weight.data = torch.tensor([[1.0]])
m1.bias.data = torch.tensor([0.0])

m2 = torch.nn.Linear(1, 1)

m2.weight.data = torch.tensor([[5.0]])
m2.bias.data = torch.tensor([3.0])

# m3 = torch.nn.Linear(1, 1)
# m3.requires_grad_(False)
x = torch.tensor([5.0])

m3 = torch.nn.Linear(1, 1)
m3.weight.data = torch.tensor([[1.0]])
m3.bias.data = torch.tensor([0.0])


sgd = torch.optim.SGD([m1.weight, m1.bias, m2.weight, m2.bias, m3.weight, m3.bias], lr=0.001)
loss_func = torch.nn.MSELoss()
target = torch.tensor([1.0])
for _ in range(1):
    y1 = m1(x)
    y2 = m2(x)
    y3 = m3(x)
    loss1 = loss_func(y1, target)
    loss2 = loss_func(y2, target)
    loss3 = loss_func(y3, target)
    loss = loss1 + loss2
    loss.backward()
    loss3.backward()
    print(f"loss1: {loss1}, loss2: {loss2}, loss3: {loss3}")
    print(f"m1.grad: {m1.weight.grad}, {m1.bias.grad}")
    # print(f"m2.grad: {m2.weight.grad}, {m2.bias.grad}")
    print(f"m3.grad: {m3.weight.grad}, {m3.bias.grad}")
    sgd.step()
print(m1.weight, m1.bias, m3.weight, m3.bias)

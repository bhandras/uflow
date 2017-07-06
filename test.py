import torch
from torch.autograd import Variable

x1 = Variable(torch.FloatTensor([2, 2, 2]), requires_grad=True)
x2 = Variable(torch.FloatTensor([3, 3, 3]), requires_grad=True)
x3 = Variable(torch.FloatTensor([4, 4, 4]), requires_grad=True)

print(x1.size())
print(x2.size())
z1 = x1 + x2
z = z1.dot(x3)

print(x1)
print(x2)
print(z)
z.backward()
print('x1.grad', x1.grad)
print('x2.grad', x2.grad)
print('x3.grad', x3.grad)

import torch
from torch.autograd import Variable

x1 = Variable(torch.FloatTensor([6]), requires_grad=True)
x2 = Variable(torch.FloatTensor([3]), requires_grad=True)
x3 = Variable(torch.FloatTensor([5]), requires_grad=True)

z = ((x1 + x2) * x2) * x3
z.backward()

print('x1.grad', x1.grad)
print('x2.grad', x2.grad)
print('x3.grad', x3.grad)


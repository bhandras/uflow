import torch
from torch.autograd import Variable
import torch.nn.functional as F

x1 = Variable(torch.FloatTensor([1, 2, 3]), requires_grad=True)
x2 = Variable(torch.FloatTensor([4, 5, 6]), requires_grad=True)

sm_x1 = F.softmax(x1)
sm_x2 = F.softmax(x2)

print(sm_x1)
print(sm_x2)

d = sm_x1.dot(sm_x2)
print('d', d)
d.backward()
print('x1 grad', x1.grad)
print('x2 grad', x2.grad)

'''
w = Variable(torch.FloatTensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))

t = w.mm(x1)
print('t = w mm x1', t)
m = x2 + t
print('m = x2 + t', m)
z = m.dot(x3)
print('z = m dot x3', z)
print(z)
z.backward()
print('x1.grad', x1.grad)
print('x2.grad', x2.grad)
print('x3.grad', x3.grad)
'''

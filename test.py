import torch
from torch.autograd import Variable

x1 = Variable(torch.FloatTensor([[2], [2], [2]]), requires_grad=True)
x2 = Variable(torch.FloatTensor([[3], [3], [3]]), requires_grad=True)
x3 = Variable(torch.FloatTensor([[4], [4], [4]]), requires_grad=True)

x1_add_x2 = x1.sub(x2)
alma = x1_add_x2.dot(x3)
print('alma', alma)
alma.backward()
print('x1 grad', x1.grad)
print('x2 grad', x2.grad)
print('x3 grad', x3.grad)

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

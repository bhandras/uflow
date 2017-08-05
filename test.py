import torch
from torch.autograd import Variable
import torch.nn.functional as F

x1 = Variable(torch.FloatTensor([[1, 2, 3]]), requires_grad=False)
x2 = Variable(torch.FloatTensor([[2, 4, 8]]), requires_grad=False)
x3 = Variable(torch.FloatTensor([[1, 2, 3], [2, 4, 8]]), requires_grad=False)

y1 = Variable(torch.LongTensor(1), requires_grad=False)
y2 = Variable(torch.LongTensor(1), requires_grad=False)
y3 = Variable(torch.LongTensor(2), requires_grad=False)

y1[0] = 1
y2[0] = 2
y3[0] = 1
y3[1] = 2

loss = torch.nn.CrossEntropyLoss()

print(loss(x1, y1))
print(loss(x2, y2))
print(loss(x3, y3))

# sm_x1 = F.softmax(x1)
# sm_x2 = F.softmax(x2)
# print(sm_x1)
# print(sm_x2)
# d = sm_x1.dot(sm_x2)
# print('d', d)
#d.backward()

# output.backward()
# print('x1 grad', x1.grad)
# print('x2 grad', x2.grad)

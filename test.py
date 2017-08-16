import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

W = Variable(torch.FloatTensor([[2, 3, 5], [7, -9, 11], [13, 17, 19]]), requires_grad=True)
b = Variable(torch.FloatTensor([3, 9, -43]), requires_grad=True)

print(W.size())
print(b.size())


m = torch.nn.LogSoftmax()
lossfn = torch.nn.NLLLoss()
x = Variable(torch.FloatTensor([[5, 99, 112], [4, 3, 7]]), requires_grad=False)
# x = Variable(torch.FloatTensor([[5, 99, 112]]), requires_grad=False)
y = Variable(torch.LongTensor([1, 2]), requires_grad=False)
# y = Variable(torch.LongTensor([1]), requires_grad=False)
print(x.size())

l1 = x.mm(W)
l1 = l1.add(b.expand_as(l1)).clamp(min=0)
l2 = l1.mm(W)
l2 = l2.add(b.expand_as(l1)).clamp(min=0)
sm = m(l2) # torch.nn.functional.log_softmax(l1)
#l1 = l1.clamp(min=0)
print(l2)
print('log_softmax', sm)

z = lossfn(sm, y)
# z = l1.dot(x)
print('loss', z)

z.backward()
print('W.grad', W.grad)
print('b.grad', b.grad)

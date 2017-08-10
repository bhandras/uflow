#include <iostream>
#include "graph.h"
#include "kernel.h"
#include "ndarray.h"
#include "mnist.h"

void test();
std::vector<VariableRef> variables;

OpRef Linear(OpRef x, size_t inp_size, size_t out_size) {
  auto W = Variable::create(x->graph(), Shape({1, inp_size, out_size}));
  variables.push_back(W);
  W->set_value(NDArray({1, inp_size, out_size},
        random_vec<float>(inp_size * out_size, 0.0f, 1.0f)));

  auto b = Variable::create(x->graph(), {1, 1, out_size});
  variables.push_back(b);
  b->set_value(NDArray({1, 1, out_size},
        random_vec<float>(out_size, 0.0f, 1.0f)));

  // std::cout << "W:\n" << W->get_value() << std::endl;
  // std::cout << "b:\n" << b->get_value() << std::endl;

  return x->bmm(W)->add(b);
}

void sgd(GraphRef g, float learning_rate) {
  for (auto& var : variables) {
    auto v = var->get_value();
    const auto& batch_grad = g->gradient(var);
    
    auto grad = batch_grad.reduce_sum(0).divs_(float(batch_grad.shape()[0]));
    v.sub_(grad.muls_(learning_rate));
    var->set_value(v);
  }
}

NDArray one_hot(size_t batch_size, size_t classes, const std::vector<int>& data) {
  NDArray result({batch_size, classes});
  for (size_t i = 0; i < data.size(); ++i) {
    result.set({i, size_t(data[i])}, 1.0f);
  }
  result.reshape({batch_size, 1, classes});
  return result;
}

// #include <xmmintrin.h>

int main() {
  // _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
  MNIST mnist;
  mnist.load("mnist", true);

  GraphRef g = std::make_shared<Graph>();
  
  auto X = Variable::create(g, {1, 28 * 28});
  auto y = Variable::create(g, {1, 10});

  auto l1 = Linear(X, 28*28, 512)->relu();
  auto l2 = Linear(l1, 512, 512)->relu();
  auto l3 = Linear(l1, 512, 10);
  auto loss = l3->softmax_ce(y);
  //auto pred = l3->softmax();
  
  size_t batch_size = 10;
  size_t classes = 10;
  int steps = 100;

  for (int i = 0; i < steps; ++i) {
    auto batch = mnist.get_train_batch(batch_size);
    X->set_value(NDArray({batch_size, 1, 28*28}, std::get<0>(batch)));
    y->set_value(one_hot(batch_size, classes, std::get<1>(batch)));
    g->forward();
    g->backward(loss);
    std::cout << "loss: " << loss->get_value() << std::endl;
    sgd(g, 0.01);
  }
  //std::cout << pred->get_value() << std::endl;
  return 0;
}


void test() {
  /*
  auto arr = NDArray({5});
  std::cout << "---" << std::endl << arr << std::endl;

  arr = NDArray({3, 2});
  std::cout << "---" << std::endl << arr << std::endl;
 
  arr = NDArray({3, 3, 4});
  std::cout << "---" << std::endl << arr << std::endl;
  
  arr = NDArray({2, 3, 3, 4});
  std::cout << "---" << std::endl << arr << std::endl; 

  NDArray a;
  a.arange(18);
  a.reshape({2, 3, 3});
  std::cout << "a=" << std::endl << a << std::endl;
  
  NDArray b;
  b.arange(18);
  b.reshape({2, 3, 3});
  std::cout << "b=" << std::endl << b << std::endl;
  std::cout << "(a dot b)= " << std::endl << a.dot(b) << std::endl;

  a.arange(4);
  std::cout << "a=" << a.str() << std::endl;
  b.arange(4);
  std::cout << "b=" << b.str() << std::endl;
  std::cout << "(a dot b)="<< a.dot(b) << std::endl;
  */

  /*
  auto zz = NDArray({2, 3}, {1, 2, 3, 4, 5, 6});
  auto qq = NDArray({2, 1}, {1, 2});
  std::cout << "QQ orig=" << qq << std::endl;
  std::cout << "QQ=" << qq.expand({2, 3}) << std::endl;
  std::cout << "zz=" << zz << std::endl;
  std::cout << "b0=" << zz.expand({2, 3}) << std::endl;
  std::cout << "b1=" << zz.expand({2, 2, 3}) << std::endl;
  std::cout << "b2=" << zz.expand({1, 2, 3}) << std::endl;
  std::cout << "b4=" << zz.expand({3, 1, 2, 2, 3}) << std::endl;
  std::cout << "zz mul qq=" << zz.mul(qq) << std::endl;
  std::cout << "qq mul zz=" << qq.mul(zz) << std::endl;
  */
  
  /*
  auto x1 = Variable::create(g, {1, 3, 1});
  auto x2 = Variable::create(g, {1, 3, 1});
  
  x1->set_value(NDArray({1, 3, 1}, {5, 7, 11}));
  x2->set_value(NDArray({1, 3, 1}, {4, 5, 6}));
  auto sm_x1 = x1->softmax();
  auto sm_x2 = x2->softmax();
  auto d = sm_x1->dot(sm_x2);
  g->eval();

  std::cout << "x1=" << std::endl << x1 << std::endl;
  std::cout << "x2=" << std::endl << x2 << std::endl;
  
  std::cout << "sm_x1=" << std::endl << sm_x1->get_value() << std::endl;
  std::cout << "sm_x2=" << std::endl << sm_x2->get_value() << std::endl;
  std::cout << "d=" << std::endl << d->get_value() << std::endl;
  std::cout << "grad x1=" << std::endl << g->gradient(x1) << std::endl;
  std::cout << "grad x2=" << std::endl << g->gradient(x2) << std::endl;
  */

  /*
  NDArray a({1, 3, 1}, {1, 2, 3});
  NDArray b({1, 3}, {4, 5, 6});
  NDArray c({2, 1, 3}, {4, 5, 6, 7, 8, 9});

  std::cout << "a=" << std::endl << a << std::endl;
  std::cout << "b=" << std::endl << b << std::endl;
  std::cout << "c=" << std::endl << c << std::endl;
  std::cout << "a mm b=" << std::endl << a.bmm(b) << std::endl;
  std::cout << "b mm a=" << std::endl << b.bmm(a) << std::endl;
  std::cout << "a bmm c=" << std::endl << a.bmm(c) << std::endl;
  std::cout << "c bmm a=" << std::endl << c.bmm(a) << std::endl;
  */

  /*
  NDArray c({2, 1, 3}, {1, 2, 3});
  NDArray d({2, 1, 3}, {4, 5, 6});

  std::cout << "c=" << c << std::endl;
  std::cout << "d=" << d << std::endl;
  std::cout << "c dot d" << c.dot(d) << std::endl;
  std::cout << "d dot c" << d.dot(c) << std::endl;

  NDArray e({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::cout << "e=" << e << std::endl;
  auto et = e.transpose();
  std::cout << "e_t=" << et << std::endl;
  */

  /* 
  NDArray xx({2}, {2, 3});
  std::cout << xx << std::endl; 
  xx.unsqueeze(1);
  std::cout << xx << std::endl; 
  
  xx = NDArray();
  xx.unsqueeze(0);
  std::cout << xx << std::endl;
  */

  /*
  NDArray x;
  x.arange(36);
  x.reshape({3, 2, 2, 3});
  std::cout << "x=" << std::endl << x << std::endl;
  std::cout << "x.sum(0)\n" << x.reduce_sum(0) << std::endl;
  std::cout << "x.sum(1)\n" << x.reduce_sum(1) << std::endl;
  std::cout << "x.sum(2)\n" << x.reduce_sum(2) << std::endl;
  std::cout << "x.sum(3)\n" << x.reduce_sum(3) << std::endl;
  std::cout << "----------------------------" << std::endl;
  
  std::cout << "x.max(0)=\n" << x.reduce_max(0) << std::endl;
  std::cout << "x.max(1)=\n" << x.reduce_max(1) << std::endl;
  std::cout << "x.max(2)=\n" << x.reduce_max(2) << std::endl;
  */
}

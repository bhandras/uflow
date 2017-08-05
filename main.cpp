#include <iostream>

#include "graph.h"
#include "kernel.h"
#include "ndarray.h"

OpRef Linear(OpRef x, const Shape& x_shape) {
  bool is_col_vec = x_shape.is_column_vector();
  size_t major = is_col_vec ? x_shape[-2] : x_shape[-1];
  
  auto W = Variable::create(x->graph(), Shape({major, major}));
  W->set_value(NDArray({major, major}, {-1, -2, -3, 4, -5, 6, 7, 8, 9}));
  auto b = Variable::create(x->graph(), is_col_vec ? Shape({major, 1}) : Shape({1, major}));

  if (is_col_vec) {
    return W->bmm(x)->add(b);
  } else {
    return x->bmm(W)->add(b);
  }
}

void test();

int main() {
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
  //return 0;
  */
  
  // test();
  // return 0;

  GraphRef g = std::make_shared<Graph>();
  
  auto X = Variable::create(g, {2, 3});
  auto y = Variable::create(g, {2, 3});

  //auto l1 = Linear(X, {2, 3, 1});
  //auto l1_relu = l1->relu();
  //auto sm = l1->softmax();
  
  auto sm = X->softmax_ce(y);
  //auto dummy = sm->dot(sm);
    
  X->set_value(NDArray({2, 3}, {1, 2, 3, 2, 4, 8}));
  y->set_value(NDArray({2, 3}, {0, 1, 0, 0, 0, 1}));
  g->eval();
  //std::cout << l1->get_value() << std::endl;
  //std::cout << l1_relu->get_value() << std::endl;
  std::cout << sm->get_value() << std::endl;
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
}

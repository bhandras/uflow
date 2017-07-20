#include "stuff.h"
#include "graph.h"
#include "kernel.h"
#include "ndarray.h"

int main() {
  Graph g;
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
  
  auto x1 = g.var(NDArray({3,1}, {2}));
  auto x2 = g.var(NDArray({3,1}, {3}));
  auto x3 = g.var(NDArray({3,1}, {4}));
  std::cout << "x1=" << x1 << std::endl;
  std::cout << "x2=" << x2 << std::endl;
  std::cout << "x3=" << x3 << std::endl;

  auto alma = g.dot(g.add(x1, x2), x3);
  g.eval();
  std::cout << "alma=" << alma << std::endl;
  std::cout << "grad x1=" << g.gradient(x1) << std::endl;
  std::cout << "grad x2=" << g.gradient(x2) << std::endl;
  std::cout << "grad x3=" << g.gradient(x3) << std::endl;
 
/*
  auto w = NDArray();
  w.arange(9);
  w.reshape({3, 3});
  auto ww = g.var(w);
  std::cout << "ww=" << ww << std::endl;
  auto t = g.mm(ww, x1);
  auto m = g.add(x2, t);
  auto z = g.dot(m, x3);
  g.eval();

  //std::cout << "m=" << m << std::endl;
  //std::cout << "z=" << z << std::endl;
  std::cout << "t=" << t->value() << std::endl;
  std::cout << "grad x1=" << g.gradient(x1) << std::endl;
  std::cout << "grad x2=" << g.gradient(x2) << std::endl;
  std::cout << "grad x3=" << g.gradient(x3) << std::endl;
  */

  
  /*
  NDArray a({3, 1}, {1, 2, 3});
  NDArray b({1, 3}, {4, 5, 6});

  std::cout << "a=" << a << std::endl;
  std::cout << "b=" << b << std::endl;
  std::cout << "a mm b=" << a.mm(b) << std::endl;
  std::cout << "b mm a=" << b.mm(a) << std::endl;

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
  return 0;
}

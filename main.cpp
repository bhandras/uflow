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

  auto x1 = g.var(NDArray({3}, {2}));
  auto x2 = g.var(NDArray({3}, {3}));
  auto x3 = g.var(NDArray({3}, {4}));
  std::cout << "x1=" << x1 << std::endl;
  std::cout << "x2=" << x2 << std::endl;
  std::cout << "x3=" << x3 << std::endl;
  auto m = g.add(x1, x2);
  auto z = g.dot(m, x3);
  g.eval();

  std::cout << "m=" << m << std::endl;
  std::cout << "z=" << z << std::endl;
  std::cout << "grad x1=" << g.gradient(x1) << std::endl;
  std::cout << "grad x2=" << g.gradient(x2) << std::endl;
  std::cout << "grad x3=" << g.gradient(x3) << std::endl;
  
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

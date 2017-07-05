#include "stuff.h"
#include "graph.h"
#include "kernel.h"
#include "ndarray.h"

int main() {
  Graph g;
  /*
  auto x1 = g.var(NDArray({6}));
  auto x2 = g.var(NDArray({3}));
  auto x3 = g.var(NDArray({5}));

  auto z = g.add(g.mul(g.mul(g.add(x1, x2), x2), x3), g.mul(x1, x3));
  g.eval();

  std::cout << x1->to_string() << ", grad: "<< g.gradient(x1) << std::endl; // 5
  std::cout << x2->to_string() << ", grad: "<< g.gradient(x2) << std::endl; // 5
  std::cout << x3->to_string() << ", grad: "<< g.gradient(x3)  << std::endl; // 9
  std::cout << "res:"<< z->to_string() << std::endl;
  auto arr = NDArray({5});
  std::cout << "---" << std::endl << arr.to_string() << std::endl;

  arr = NDArray({3, 2});
  std::cout << "---" << std::endl << arr.to_string() << std::endl;
 
  arr = NDArray({3, 3, 4});
  std::cout << "---" << std::endl << arr.to_string() << std::endl;
  
  arr = NDArray({2, 3, 3, 4});
  std::cout << "---" << std::endl << arr.to_string() << std::endl; 
  */

  NDArray a;
  a.arange(18);
  a.reshape({2, 3, 3});
  std::cout << "a=" << std::endl << a.to_string() << std::endl;
  
  NDArray b;
  b.arange(18);
  b.reshape({2, 3, 3});
  std::cout << "b=" << std::endl << b.to_string() << std::endl;
 
  auto c = a.dot(b);
  std::cout << "(a dot b)= " << std::endl << c.to_string() << std::endl;

  a.arange(4);
  std::cout << "a=" << a.to_string() << std::endl;
  b.arange(4);
  std::cout << "b=" << b.to_string() << std::endl;
  std::cout << "(a dot b)="<< a.dot(b).to_string() << std::endl;
  return 0;
}

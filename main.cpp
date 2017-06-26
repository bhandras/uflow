#include "stuff.h"
#include "graph.h"
#include "kernel.h"

int main() {
  Graph g;
  auto x1 = g.var(3);
  auto x2 = g.var(4);
  auto x3 = g.var(9);

  auto z = g.mul(g.add(x1, x2), x3);
  g.eval();

  std::cout << x1->value() << ", grad: "<< g.gradient(x1) << std::endl;
  std::cout << x2->value() << ", grad: "<< g.gradient(x2) << std::endl;
  std::cout << x3->value() << ", grad: "<< g.gradient(x3)  << std::endl;
  std::cout << z->value() << ", grad: "<< g.gradient(z) << std::endl;
  
  return 0;
}

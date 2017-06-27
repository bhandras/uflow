#include "stuff.h"
#include "graph.h"
#include "kernel.h"

int main() {
  Graph g;
  auto x1 = g.var(6);
  auto x2 = g.var(3);
  auto x3 = g.var(5);

  auto z = g.mul(g.mul(g.add(x1, x2), x2), x3);
  g.eval();

  std::cout << x1->to_string() << ", grad: "<< g.gradient(x1) << std::endl; // 5
  std::cout << x2->to_string() << ", grad: "<< g.gradient(x2) << std::endl; // 5
  std::cout << x3->to_string() << ", grad: "<< g.gradient(x3)  << std::endl; // 9
  std::cout << "res:"<< z->to_string() << std::endl;
  
  return 0;
}

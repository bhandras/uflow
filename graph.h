#ifndef _graph_h_
#define _graph_h_

#include <list>
#include <queue>
#include <vector>
#include <memory>
#include <ostream>
#include <unordered_map>

#include "ndarray.h"

class Kernel;

class Node {
  public:
    typedef std::shared_ptr<Node> ptr;

    Node(std::unique_ptr<Kernel> kernel);
    virtual ~Node() {}
    Kernel& kernel() { return *kernel_.get(); } 
    void eval();
    const NDArray& value() const;
    std::string str() const;

  protected:
    std::unique_ptr<Kernel> kernel_;
};

class Graph {
  public:
    Node::ptr var(NDArray value);
    Node::ptr add(Node::ptr a, Node::ptr b);
    Node::ptr sub(Node::ptr a, Node::ptr b);
    Node::ptr mul(Node::ptr a, Node::ptr b);
    Node::ptr dot(Node::ptr a, Node::ptr b);
    Node::ptr mm(Node::ptr a, Node::ptr b);
    void eval();
    NDArray gradient(const Node::ptr& node) const;

  protected:
    std::unordered_map<Node::ptr, std::list<Node::ptr>> adj_;
    std::unordered_map<Node::ptr, NDArray> gradients_;
};

std::ostream& operator<<(std::ostream& os, const Node& node);
std::ostream& operator<<(std::ostream& os, const Node::ptr& node);
 
#endif // _graph_h_


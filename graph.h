#ifndef _graph_h_
#define _graph_h_

#include <list>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>

#include "ndarray.h"

class Kernel;

class Node {
  public:
    typedef std::shared_ptr<Node> ptr;

    Node(std::unique_ptr<Kernel> kernel);
    virtual ~Node() {}
    
    void eval();
    const NDArray& value() const;
    NDArray gradient(const Node::ptr& node);
    std::string to_string() const;

  protected:
    std::unique_ptr<Kernel> kernel_;
};


class Graph {
  public:
    Node::ptr var(NDArray value);
    Node::ptr add(Node::ptr a, Node::ptr b);
    Node::ptr mul(Node::ptr a, Node::ptr b);
    Node::ptr dot(Node::ptr a, Node::ptr b);
    void eval();
    NDArray gradient(const Node::ptr& node) const;

  protected:
    std::unordered_map<Node::ptr, std::list<Node::ptr>> adj_;
    std::unordered_map<Node::ptr, NDArray> gradients_;
};


#endif // _graph_h_


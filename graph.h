#ifndef _graph_h_
#define _graph_h_

#include <list>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>

class Kernel;

class Node {
  public:
    typedef std::shared_ptr<Node> ptr;

    Node(std::unique_ptr<Kernel> kernel);
    virtual ~Node() {}
    
    void eval();
    float value() const;
    float gradient(const Node::ptr& node);

  protected:
    std::unique_ptr<Kernel> kernel_;
};


class Graph {
  public:
    Node::ptr var(float value);
    Node::ptr add(Node::ptr a, Node::ptr b);
    Node::ptr mul(Node::ptr a, Node::ptr b);
    void eval();
    float gradient(const Node::ptr& node) const;

  protected:
    std::unordered_map<Node::ptr, std::list<Node::ptr>> adj_;
    std::unordered_map<Node::ptr, float> gradients_;
};


#endif // _graph_h_


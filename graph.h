#ifndef _graph_h_
#define _graph_h_

#include <list>
#include <vector>
#include <memory>
#include <ostream>
#include <unordered_map>

#include "ndarray.h"


class Node;
class Graph;
class Op;
class Variable;
class Kernel;
class ValueKernel;
typedef std::shared_ptr<Node> NodeRef;
typedef std::shared_ptr<Graph> GraphRef;
typedef std::shared_ptr<Op> OpRef;
typedef std::shared_ptr<Variable> VariableRef;
typedef std::shared_ptr<Kernel> KernelRef;


class Node : public std::enable_shared_from_this<Node> {
  public:
    virtual ~Node();   
    
    NodeRef ref();
    GraphRef graph();

    virtual KernelRef kernel() const = 0;
    virtual bool requires_grad() const = 0;

    const NDArray& get_value() const;
    virtual std::string str() const;

  protected:
    Node(GraphRef g);
    Node(const Node&) = delete;

    GraphRef graph_;
};


class Op : public Node {
  protected:
    struct protected_;

  public:
    explicit Op(const protected_&, GraphRef graph);
    virtual ~Op();

    OpRef ref();
    operator NodeRef();

    OpRef add(NodeRef other);
    OpRef sub(NodeRef other);
    OpRef mul(NodeRef other);
    OpRef dot(NodeRef other);
    OpRef mm(NodeRef other);
    OpRef softmax();

    virtual std::string str() const override;

  protected:
    virtual KernelRef kernel() const override {
      return kernel_;
    }
    virtual bool requires_grad() const override {
      return false;
    }

    struct protected_ {
      explicit protected_(int) { }
    };

    
  private:
    Op(const Op&) = delete;
    const Op& operator=(const Op&) = delete;
    
    void set_kernel(KernelRef kernel) {
      kernel_ = kernel;
    }

    template <class K, class... Args>
    OpRef op(const std::string& name, Args... args);

    KernelRef kernel_;
};

class Variable : public Op {
 public:
    static VariableRef create(GraphRef graph,
        const std::vector<size_t>& shape, bool requires_grad=true);

    Variable(const Op::protected_&, GraphRef graph,
        const std::vector<size_t>& shape, bool requires_grad);

    virtual ~Variable();
  
    VariableRef ref();
    operator NodeRef();

    void set_value(const NDArray& value);
    virtual std::string str() const override;

  protected:
    virtual bool requires_grad() const override;
    virtual KernelRef kernel() const override;

  private:
    Variable() = delete;
    Variable(const Variable&) = delete;
    const Variable& operator=(const Variable&) = delete;

    std::shared_ptr<ValueKernel> kernel_;
    std::vector<size_t> shape_;
    bool requires_grad_;
};


class Graph {
  public:
    void add(NodeRef a);
    void eval();
    NDArray gradient(const NodeRef& node) const;

  protected:
    std::unordered_map<NodeRef, std::list<NodeRef>> adj_;
    std::unordered_map<NodeRef, NDArray> gradients_;
};

std::ostream& operator<<(std::ostream& os, const NodeRef& node);
std::ostream& operator<<(std::ostream& os, const OpRef& node);
std::ostream& operator<<(std::ostream& os, const VariableRef& node);
 
#endif // _graph_h_


#ifndef _kernel_h_
#define _kernel_h_

#include <list>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>

#include "ndarray.h"
#include "graph.h"

class Kernel {
  public:
    Kernel()
      : value_({1}) { }

    Kernel(NDArray value)
      : value_(value) { }
    
    virtual ~Kernel() { }
    virtual void forward() { }

    virtual const NDArray& value() const {
      return value_;
    }

    virtual void backward(const NDArray& suc_gradient,
        std::unordered_map<Node::ptr, NDArray>& gradients) const { }

    virtual std::string str() const {
      return std::string();
    }

  protected:
    NDArray value_;
};


class Value : public Kernel {
  public:
    Value(NDArray value)
      : Kernel(value) { }

    virtual std::string str() const override;
};


class Add : public Kernel {
  public:
    Add(Node::ptr a, Node::ptr b);
    
    virtual void forward() override;
    virtual void backward(const NDArray& suc_gradient,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;

  private:
    Node::ptr a_;
    Node::ptr b_;
};


class Sub : public Kernel {
  public:
    Sub(Node::ptr a, Node::ptr b);
    virtual void forward() override;
    virtual void backward(const NDArray& suc_gradient,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;
  private:
    Node::ptr a_;
    Node::ptr b_;
};


class Mul : public Kernel {
  public:
    Mul(Node::ptr a, Node::ptr b);

    virtual void forward() override;
    virtual void backward(const NDArray& suc_gradient,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;

  private:
    Node::ptr a_;
    Node::ptr b_;
};


class Dot : public Kernel {
  public:
    Dot(Node::ptr a, Node::ptr b);
    
    virtual void forward() override;
    virtual void backward(const NDArray& suc_gradient,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;

  private:
    Node::ptr a_;
    Node::ptr b_;
};


class MatMul : public Kernel {
  public:
    MatMul(Node::ptr a, Node::ptr b);
    
    virtual void forward() override;
    virtual void backward(const NDArray& suc_gradient,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;

  private:
    Node::ptr a_;
    Node::ptr b_;
};

class Softmax : public Kernel {
  public:
    Softmax(Node::ptr node);
    virtual void forward() override;
    virtual void backward(const NDArray& suc_gradient,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;
    virtual std::string str() const override;

  private:
    Node::ptr node_;
    NDArray derivative_;
};

#endif // _kernel_h_


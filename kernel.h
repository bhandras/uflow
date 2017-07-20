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

    virtual void backward(const std::list<Node::ptr>& outputs,
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
    Add(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
    
    virtual void forward() override;
    virtual void backward(const std::list<Node::ptr>& outputs,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;

  private:
    std::shared_ptr<Node> a_;
    std::shared_ptr<Node> b_;
};


class Mul : public Kernel {
  public:
    Mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);

    virtual void forward() override;
    virtual void backward(const std::list<Node::ptr>& outputs,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;

  private:
    std::shared_ptr<Node> a_;
    std::shared_ptr<Node> b_;
};


class Dot : public Kernel {
  public:
    Dot(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
    
    virtual void forward() override;
    virtual void backward(const std::list<Node::ptr>& outputs,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;

  private:
    std::shared_ptr<Node> a_;
    std::shared_ptr<Node> b_;
};


class MatMul : public Kernel {
  public:
    MatMul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
    
    virtual void forward() override;
    virtual void backward(const std::list<Node::ptr>& outputs,
        std::unordered_map<Node::ptr, NDArray>& gradients) const override;

    virtual std::string str() const override;

  private:
    std::shared_ptr<Node> a_;
    std::shared_ptr<Node> b_;
};



#endif // _kernel_h_


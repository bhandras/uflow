#ifndef _kernel_h_
#define _kernel_h_

#include <vector>
#include <memory>
#include <unordered_map>

#include "ndarray.h"
#include "graph.h"

class Kernel {
  public:
    virtual ~Kernel() { }
    virtual void forward() { }
    virtual void backward(const NDArray& output_grad) { }
  
    const NDArray& get_value() const {
      return value_;
    }
    
    void set_inputs(std::vector<NodeRef>& inputs) {
      inputs_.clear();
      inputs_.insert(inputs_.end(), inputs.begin(), inputs.end());
    }

    const std::vector<NodeRef>& get_inputs() const {
      return inputs_;
    }

    const NDArray& get_gradient(const NodeRef& node) const {
      static NDArray default_grad({1}, {0});

      const auto& it = gradients_.find(node);
      if (it != gradients_.end()) {
        return it->second;
      }

      return default_grad;
    }

    void clear_gradients() {
      gradients_.clear();
    }

    virtual std::string str() const {
      return "kernel";
    }

  protected:
    NDArray value_;
    std::vector<NodeRef> inputs_;
    std::unordered_map<NodeRef, NDArray> gradients_;
};

class ValueKernel : public Kernel {
  public:
    ValueKernel(const Shape& shape) {
      // TODO
      value_.zeros(shape.v());
    }
    void set_value(const NDArray& value) {
      value_ = value;
    }

    virtual std::string str() const {
      return value_.str();
    }
};

class AddKernel : public Kernel {
  public:
    AddKernel() = default;
    virtual std::string str() const override;
    
  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;
};


class SubKernel : public Kernel {
  public:
    SubKernel() = default;
    virtual std::string str() const override;

  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;
};


class MulKernel : public Kernel {
  public:
    MulKernel() = default;
    virtual std::string str() const override;

  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;
};


class DotKernel : public Kernel {
  public:
    DotKernel() = default;
    virtual std::string str() const override;

  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;
};


class MatMulKernel : public Kernel {
  public:
    MatMulKernel() = default;
    virtual std::string str() const override;

  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;
};


class BatchMatMulKernel : public Kernel {
  public:
    BatchMatMulKernel() = default;
    virtual std::string str() const override;

  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;
};


class SoftmaxKernel : public Kernel {
  public:
    SoftmaxKernel() = default;
    virtual std::string str() const override;

  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;

  private:
    NDArray derivative_;
};

class SoftmaxCrossEntropyKernel : public Kernel {
  public:
    SoftmaxCrossEntropyKernel() = default;
    virtual std::string str() const override;

  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;

  private:
    NDArray derivative_;
};

class ReLUKernel : public Kernel {
  public:
    ReLUKernel() = default;
    virtual std::string str() const override;

  protected:
    virtual void forward() override;
    virtual void backward(const NDArray& output_grad) override;
 
  private:
    NDArray derivative_;
};

#endif // _kernel_h_


#ifndef _kernel_h_
#define _kernel_h_

#include <list>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>

class Node;

class Kernel {
  public:
    Kernel()
      : value_(0.0f) { }

    Kernel(float value)
      : value_(value) { }
    
    virtual ~Kernel() { }
    virtual void forward() { }

    virtual float value() const {
      return value_;
    }

    virtual float gradient(const std::shared_ptr<Node>& node) {
      return 0.0f;
    }

    virtual std::string to_string() const {
      return std::string();
    }

  protected:
    float value_;
};


class Value : public Kernel {
  public:
    Value(float value)
      : Kernel(value) { }

    virtual std::string to_string() const override;
};

class Add : public Kernel {
  public:
    Add(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
    
    virtual void forward() override;
    virtual float gradient(const std::shared_ptr<Node>& node) override;
    virtual std::string to_string() const override;

  private:
    std::shared_ptr<Node> a_;
    std::shared_ptr<Node> b_;
};


class Mul : public Kernel {
  public:
    Mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);

    virtual void forward() override;
    float gradient(const std::shared_ptr<Node>& node) override;
    virtual std::string to_string() const override;

  private:
    std::shared_ptr<Node> a_;
    std::shared_ptr<Node> b_;
};


#endif // _kernel_h_


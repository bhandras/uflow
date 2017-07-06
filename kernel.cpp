#include <string>
#include "kernel.h"
#include "graph.h"
    

std::string Value::str() const {
  return value_.str();
}

Add::Add(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Add::forward() {
  value_ = a_->value().add(b_->value());
}

NDArray Add::gradient(const std::shared_ptr<Node>& node)  {
  NDArray zeros(a_->value().shape(), {0});
  if (node != a_ && node != b_) return zeros;
  NDArray ones(a_->value().shape(), {1});
  return ones;
}

std::string Add::str() const {
  return "(" + a_->value().str() + " + " + b_->value().str() +  ")";
}

Mul::Mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Mul::forward() {
  value_ = a_->value().mul(b_->value());
}

NDArray Mul::gradient(const std::shared_ptr<Node>& node) {
  if (node == a_) return b_->value();
  if (node == b_) return a_->value();
  NDArray zero(a_->value().shape(), {0});
  return zero;
}

std::string Mul::str() const {
  return "(" + a_->value().str() + " * " + b_->value().str() + ")";
}

Dot::Dot(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Dot::forward() {
  value_ = a_->value().dot(b_->value());
}

NDArray Dot::gradient(const std::shared_ptr<Node>& node) {
  if (node == a_) return b_->value();
  if (node == b_) return a_->value();
  NDArray zero(a_->value().shape(), {0});
  return zero;
}

std::string Dot::str() const {
  return "(" + a_->value().str() + " dot " + b_->value().str() + ")";
}



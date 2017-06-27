#include <string>
#include "kernel.h"
#include "graph.h"
    

std::string Value::to_string() const {
  return "(val: " + std::to_string(value_) + ")";
}

Add::Add(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Add::forward() {
  value_ = a_->value() + b_->value();
}

float Add::gradient(const std::shared_ptr<Node>& node)  {
  if (node != a_ && node != b_) return 0.0f;
  return 1.0f;
}

std::string Add::to_string() const {
  return "(" + std::to_string(a_->value()) + " + " + std::to_string(b_->value()) +  ")";
}

Mul::Mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Mul::forward() {
  value_ = a_->value() * b_->value();
}

float Mul::gradient(const std::shared_ptr<Node>& node) {
  if (node == a_) return b_->value();
  if (node == b_) return a_->value();
  return 0.0f;
}

std::string Mul::to_string() const {
  return "(" + std::to_string(a_->value()) + " * " + std::to_string(b_->value()) + ")";
}



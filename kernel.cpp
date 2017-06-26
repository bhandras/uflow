#include "kernel.h"
#include "graph.h"

Add::Add(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Add::forward() {
  value_ = a_->value() + b_->value();
}

float Add::gradient(const std::shared_ptr<Node>& node)  {
  if (node != a_ && node != b_) return 0.0f;
  return 1.0f;
}

Mul::Mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Mul::forward() {
  value_ = a_->value() * b_->value();
}
#include <iostream>
float Mul::gradient(const std::shared_ptr<Node>& node) {
  if (node == a_) return b_->value();
  if (node == b_) return a_->value();
  return 0.0f;
}


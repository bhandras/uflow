#include <iostream>
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

void Add::backward(const std::list<Node::ptr>& outputs,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  gradients[a_] = NDArray(a_->value().shape());
  
  for (auto& output : outputs) {
    gradients[a_].add_(output->value());
  }

  gradients[b_] = gradients[a_];
}

std::string Add::str() const {
  return "(" + a_->value().str() + " + " + b_->value().str() +  ")";
}

Mul::Mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Mul::forward() {
  value_ = a_->value().mul(b_->value());
}

void Mul::backward(const std::list<Node::ptr>& outputs,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  gradients[a_] = NDArray(a_->value().shape());
  gradients[b_] = NDArray(a_->value().shape());
  
  for (auto& output : outputs) {
    gradients[a_].add_(b_->value().mul(output->value()));
    gradients[b_].add_(a_->value().mul(output->value()));
  }
}

std::string Mul::str() const {
  return "(" + a_->value().str() + " * " + b_->value().str() + ")";
}

Dot::Dot(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Dot::forward() {
  value_ = a_->value().dot(b_->value());
}

void Dot::backward(const std::list<Node::ptr>& outputs,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  std::cout << "dot" << std::endl;
  gradients[a_] = NDArray(a_->value().shape());
  gradients[b_] = NDArray(a_->value().shape());
  
  for (auto& output : outputs) {
    gradients[a_].add_(output->value().dot(b_->value()));
    gradients[b_].add_(a_->value().dot(output->value()));
  }
}

std::string Dot::str() const {
  return "(" + a_->value().str() + " dot " + b_->value().str() + ")";
}

MatMul::MatMul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void MatMul::forward() {
  value_ = a_->value().mm(b_->value());
}

void MatMul::backward(const std::list<Node::ptr>& outputs,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  std::cout << "matmul" << std::endl;
  gradients[a_] = NDArray(a_->value().shape());
  gradients[b_] = NDArray(a_->value().shape());
  auto a_t = a_->value().transpose();
  auto b_t = b_->value().transpose();
  
  for (auto& output : outputs) {
    gradients[a_].add_(output->value().mm(b_t));
    gradients[b_].add_(a_t.mm(output->value()));
  }
}

std::string MatMul::str() const {
  return "(" + a_->value().str() + " mm " + b_->value().str() + ")";
}


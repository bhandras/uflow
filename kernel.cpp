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

void Add::backward(const NDArray& suc_gradient,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  if (gradients.empty()) {
    gradients[a_] = NDArray(a_->value().shape());
    gradients[b_] = NDArray(b_->value().shape());
  }
  
  gradients[a_].add_(suc_gradient);
  gradients[b_].add_(suc_gradient);
}

std::string Add::str() const {
  return "(" + a_->value().str() + " + " + b_->value().str() +  ")";
}


Sub::Sub(Node::ptr a, Node::ptr b)
  : a_(a), b_(b) { }

void Sub::forward() {
  value_ = a_->value().sub(b_->value());
}

void Sub::backward(const NDArray& suc_gradient,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  if (gradients.empty()) {
    gradients[a_] = NDArray(a_->value().shape());
    gradients[b_] = NDArray(b_->value().shape());
  }
  
  gradients[a_].add_(suc_gradient);
  gradients[b_].sub_(suc_gradient);
}

std::string Sub::str() const {
  return "(" + a_->value().str() + " - " + b_->value().str() +  ")";
}


Mul::Mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Mul::forward() {
  value_ = a_->value().mul(b_->value());
}

void Mul::backward(const NDArray& suc_gradient,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  if (gradients.empty()) {
    gradients[a_] = NDArray(b_->value().shape());
    gradients[b_] = NDArray(a_->value().shape());
  }

  gradients[a_].add_(b_->value().mul(suc_gradient));
  gradients[b_].add_(a_->value().mul(suc_gradient));
}

std::string Mul::str() const {
  return "(" + a_->value().str() + " * " + b_->value().str() + ")";
}

Dot::Dot(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void Dot::forward() {
  value_ = a_->value().dot(b_->value());
}

void Dot::backward(const NDArray& suc_gradient,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  if (gradients.empty()) {
    gradients[a_] = NDArray(b_->value().shape());
    gradients[b_] = NDArray(a_->value().shape());
  }
  gradients[a_].add_(suc_gradient.dot(b_->value()));
  gradients[b_].add_(a_->value().dot(suc_gradient));
}

std::string Dot::str() const {
  return "(" + a_->value().str() + " dot " + b_->value().str() + ")";
}

MatMul::MatMul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
  : a_(a), b_(b) { }

void MatMul::forward() {
  value_ = a_->value().mm(b_->value());
}

void MatMul::backward(const NDArray& suc_gradient,
    std::unordered_map<Node::ptr, NDArray>& gradients) const {
  if (gradients.empty()) {
    gradients[a_] = NDArray(b_->value().shape());
    gradients[b_] = NDArray(a_->value().shape());
  }

  auto a_t = a_->value().transpose();
  auto b_t = b_->value().transpose();

  gradients[a_].add_(suc_gradient.mm(b_t));
  gradients[b_].add_(a_t.mm(suc_gradient));
}

std::string MatMul::str() const {
  return "(" + a_->value().str() + " mm " + b_->value().str() + ")";
}


#include <string>
#include "kernel.h"
#include "graph.h"

void AddKernel::forward() {
  value_ = inputs_[0]->get_value().add(inputs_[1]->get_value());
}

void AddKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(inputs_[0]->get_value().shape());
    gradients_[inputs_[1]].zeros(inputs_[1]->get_value().shape());
  }
  
  gradients_[inputs_[0]].add_(output_grad);
  gradients_[inputs_[1]].add_(output_grad);
}

std::string AddKernel::str() const {
  return "("
    + inputs_[0]->get_value().str()
    + " + "
    + inputs_[1]->get_value().str()
    +  ")";
}

void SubKernel::forward() {
  value_ = inputs_[0]->get_value().sub(inputs_[1]->get_value());
}

void SubKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(inputs_[0]->get_value().shape());
    gradients_[inputs_[1]].zeros(inputs_[1]->get_value().shape());
  }
  
  gradients_[inputs_[0]].add_(output_grad);
  gradients_[inputs_[1]].sub_(output_grad);
}

std::string SubKernel::str() const {
  return "("
    + inputs_[0]->get_value().str()
    + " - "
    + inputs_[1]->get_value().str()
    +  ")";
}


void MulKernel::forward() {
  value_ = inputs_[0]->get_value().mul(inputs_[1]->get_value());
}

void MulKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(inputs_[1]->get_value().shape());
    gradients_[inputs_[1]].zeros(inputs_[0]->get_value().shape());
  }

  gradients_[inputs_[0]].add_(inputs_[1]->get_value().mul(output_grad));
  gradients_[inputs_[1]].add_(inputs_[0]->get_value().mul(output_grad));
}

std::string MulKernel::str() const {
  return "("
    + inputs_[0]->get_value().str()
    + " * "
    + inputs_[1]->get_value().str()
    + ")";
}


void DotKernel::forward() {
  value_ = inputs_[0]->get_value().dot(inputs_[1]->get_value());
}

void DotKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(inputs_[1]->get_value().shape());
    gradients_[inputs_[1]].zeros(inputs_[0]->get_value().shape());
  }

  gradients_[inputs_[0]].add_(output_grad.dot(inputs_[1]->get_value()));
  gradients_[inputs_[1]].add_(inputs_[0]->get_value().dot(output_grad));
}

std::string DotKernel::str() const {
  return "("
    + inputs_[0]->get_value().str()
    + " dot "
    + inputs_[1]->get_value().str()
    + ")";
}


void MatMulKernel::forward() {
  value_ = inputs_[0]->get_value().mm(inputs_[1]->get_value());
}

void MatMulKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(inputs_[1]->get_value().shape());
    gradients_[inputs_[1]].zeros(inputs_[0]->get_value().shape());
  }

  auto a_t = inputs_[0]->get_value().transpose();
  auto b_t = inputs_[1]->get_value().transpose();

  gradients_[inputs_[0]].add_(output_grad.mm(b_t));
  gradients_[inputs_[1]].add_(a_t.mm(output_grad));
}

std::string MatMulKernel::str() const {
  return "("
    + inputs_[0]->get_value().str()
    + " mm "
    + inputs_[1]->get_value().str()
    + ")";
}

// Great explanation: 
// http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
void SoftmaxKernel::forward() {
  value_ = inputs_[0]->get_value();
  
  auto m = value_.max();
  value_.sub_(m).exp_();
  value_.mul_(value_.sum().recip_());

  // todo: at the moment we suppose that value_ is a row vector
  size_t d1 = value_.shape()[0];
  size_t d2 = value_.shape()[1];
  // todo: handle batches
  // derivative_.zeros({d1, d2, d2});
  derivative_.zeros({d2, d2});

  for (size_t d = 0; d < d1; ++d) {
    for (size_t i = 0; i < d2; ++i) {
      for (size_t j = 0; j < d2; ++j) {
        if (i == j) {
          float v_i = value_.get({d, i});
          derivative_.set({i, j}, v_i * (1.0f - v_i));
        } else {
          float v_i = value_.get({d, i});
          float v_j = value_.get({d, j});
          derivative_.set({i, j}, -v_i * v_j);
        }
      }
    }
  }
}

void SoftmaxKernel::backward(const NDArray& output_grad) {
  NDArray og = output_grad;
  // todo: handle batches
  // og.unsqueeze(0);

  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(og.shape());
  }
  gradients_[inputs_[0]].add_(og.mm(derivative_));
}

std::string SoftmaxKernel::str() const {
  return "softmax("
    + inputs_[0]->get_value().str()
    + ")";
}


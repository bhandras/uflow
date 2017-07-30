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
  
  Shape shape(value_.shape());
  bool invalid = false;

  if (shape.size() != 2 && shape.size() != 3) {
    invalid = true;
  }

  bool is_row_vec = shape.is_row_vector();
  bool is_col_vec = shape.is_column_vector();

  if (!is_row_vec && !is_col_vec) {
    invalid = true;
  }

  if (invalid) {
    throw ValueError("Incompatible shape for softmax: " + vstr(shape.v()));
  }

  // make sure we're dealing with column vectors
  if (is_row_vec) {
    shape.swap(-2, -1);
    value_.reshape(shape.v());
  }

  auto m = value_.max();
  value_.sub_(m).exp_();
  value_.mul_(value_.sum().recip_());

  // number of batches
  size_t nb = shape.size() == 3 ? shape[0] : 0;
  size_t count = nb > 0 ? nb : 1;
  size_t js = shape[-2]; // size of the Jacobian

  derivative_.zeros({count, js, js});
  if (nb == 0) {
    value_.unsqueeze(0);
  }

  for (size_t b = 0; b < count; ++b) { // batch
    for (size_t i = 0; i < js; ++i) { // Jacobian row
      for (size_t j = 0; j < js; ++j) { // Jacobian col
        if (i == j) {
          float v_i = value_.get({b, i, 0});
          float val = v_i * (1.0f - v_i);
          derivative_.set({b, i, i}, val);
        } else {
          float v_i = value_.get({b, i, 0});
          float v_j = value_.get({b, j, 0});
          derivative_.set({b, i, j}, -v_i * v_j);
        }
      }
    }
  }

  if (nb == 0) {
    derivative_.squeeze(0);
    value_.squeeze(0);
  }

  if (is_row_vec) {
    shape.swap(-1, -2);
    value_.reshape(shape.v());
  }
}

void SoftmaxKernel::backward(const NDArray& output_grad) {
  Shape og_shape(output_grad.shape());

  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(output_grad.shape());
  }

  if (og_shape.is_row_vector()) {
    gradients_[inputs_[0]].add_(output_grad.mm(derivative_));
  } else {
    gradients_[inputs_[0]].add_(derivative_.mm(output_grad));
  }
}

std::string SoftmaxKernel::str() const {
  return "softmax("
    + inputs_[0]->get_value().str()
    + ")";
}


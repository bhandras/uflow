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


void BatchMatMulKernel::forward() {
  value_ = inputs_[0]->get_value().bmm(inputs_[1]->get_value());
}

void BatchMatMulKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(inputs_[1]->get_value().shape());
    gradients_[inputs_[1]].zeros(inputs_[0]->get_value().shape());
  }

  auto a_t = inputs_[0]->get_value().transpose();
  auto b_t = inputs_[1]->get_value().transpose();

  gradients_[inputs_[0]].add_(output_grad.bmm(b_t));
  gradients_[inputs_[1]].add_(a_t.bmm(output_grad));
}

std::string BatchMatMulKernel::str() const {
  return "("
    + inputs_[0]->get_value().str()
    + " bmm "
    + inputs_[1]->get_value().str()
    + ")";
}

// Great explanation: 
// http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
void SoftmaxKernel::forward() {
  value_ = inputs_[0]->get_value();
  
  Shape shape(value_.shape());
  bool invalid = false;

  bool unsqueezed = false; 
  if (shape.size() != 3) {
    if (shape.size() == 2) {
      value_.unsqueeze(2);
      unsqueezed = true;
    } else {
      invalid = true;
    }
  }
  
  if (!invalid && shape[-1] != 1 && shape[-2] != 1) {
    // can only softmax(_, m, n) if m == 1 or n == 1
    invalid = true;
  }

  if (invalid) {
    throw ValueError("Incompatible shape for softmax: " + vstr(shape.v()));
  }

  bool swapped = false;
  // make sure we're dealing with column vectors
  if (shape[-1] != 1) {
    shape.swap(-2, -1);
    value_.reshape(shape.v());
  }

  auto m = value_.reduce_max(1, true);
  value_.sub_(m).exp_();
  value_.mul_(value_.reduce_sum(1, true).recip_());

  size_t nb = shape[0]; // number of batches
  size_t js = shape[-2]; // size of the Jacobian

  derivative_.zeros({nb, js, js});

  for (size_t b = 0; b < nb; ++b) { // batch
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

  if (swapped) {
    shape.swap(-1, -2);
    value_.reshape(shape.v());
  }

  if (unsqueezed) {
    value_.squeeze(2);
  }
}

void SoftmaxKernel::backward(const NDArray& output_grad) {
  Shape og_shape(output_grad.shape());

  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(output_grad.shape());
  }

  if (og_shape.is_row_vector()) {
    gradients_[inputs_[0]].add_(output_grad.bmm(derivative_));
  } else {
    gradients_[inputs_[0]].add_(derivative_.bmm(output_grad));
  }
}

std::string SoftmaxKernel::str() const {
  return "softmax("
    + inputs_[0]->get_value().str()
    + ")";
}

void SoftmaxCrossEntropyKernel::forward() {
  // TODO: copy on write...
  auto x = inputs_[0]->get_value();
  auto y = inputs_[1]->get_value();

  if (x.shape() != y.shape()) {
    throw ValueError("Incompatible input and target shapes for softmax CE: "
        + vstr(x.shape())
        + ", "
        + vstr(y.shape()));
  }

  Shape shape(x.shape());
  bool invalid = false;

  bool unsqueezed = false; 
  if (shape.size() != 3) {
    if (shape.size() == 2) {
      x.unsqueeze(2);
      y.unsqueeze(2);
      shape.unsqueeze(2);
      unsqueezed = true;
    } else {
      invalid = true;
    }
  }
  
  if (!invalid && shape[-1] != 1 && shape[-2] != 1) {
    // can only softmax(_, m, n) if m == 1 or n == 1
    invalid = true;
  }

  if (invalid) {
    throw ValueError("Incompatible shape for softmax: " + vstr(shape.v()));
  }

  value_ = x;

  bool swapped = false;
  // make sure we're dealing with column vector
  if (shape[-1] != 1) {
    shape.swap(-2, -1);
    x.reshape(shape.v());
    y.reshape(shape.v());
    swapped = true;
  }

  auto max_x = x.reduce_max(1, true);
  x.sub_(max_x).exp_();
  x.mul_(x.reduce_sum(1, true).recip_());
  auto p = x; // p = softmax(x)

  // calculate CE loss
  x.log_().mul_(NDArray({1}, {-1}).mul_(y));
  value_ = x.reduce_sum(1, false).reduce_sum();
  value_.mul_(NDArray({1}, {1.0f / shape[0]}));
  // std::cout << "ce=\n" << value_ << std::endl;
 
  // calculaet derivative
  derivative_ = p.sub_(y);

  if (swapped) {
    shape.swap(-2, -1);
    derivative_.reshape(shape.v());
  }
  
  if (unsqueezed) {
    derivative_.squeeze(2);
  }
}

void SoftmaxCrossEntropyKernel::backward(const NDArray& output_grad) {
  Shape og_shape(output_grad.shape());

  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(output_grad.shape());
  }

  gradients_[inputs_[0]].add_(output_grad.mul(derivative_));
}

std::string SoftmaxCrossEntropyKernel::str() const {
  return "softmax CE("
    + inputs_[0]->get_value().str()
    + ")";
}

std::string ReLUKernel::str() const {
  return "ReLU("
    + inputs_[0]->get_value().str()
    + ")";
}

void ReLUKernel::forward() {
  value_ = inputs_[0]->get_value().max_filter(0.0f);
}

void ReLUKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]] = value_;
  }

  gradients_[inputs_[0]].clip_(0.0f, 1.0f);
}


#include <string>
#include "kernel.h"
#include "graph.h"

void AddKernel::forward() {
  value_ = inputs_[0]->get_value().add(inputs_[1]->get_value());
}

void AddKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]] = output_grad;
    gradients_[inputs_[1]] = output_grad.reduce_sum(0, false);
  } else {
    gradients_[inputs_[0]].add_(output_grad);
    gradients_[inputs_[1]].add_(output_grad.reduce_sum(0, false));
  }
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
    gradients_[inputs_[0]].zeros(inputs_[0]->get_value().shape());
    gradients_[inputs_[1]].zeros(inputs_[1]->get_value().shape());
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
  auto a_t = inputs_[0]->get_value().transpose();
  auto b_t = inputs_[1]->get_value().transpose();
  auto g0 = output_grad.bmm(b_t);
  auto g1 = a_t.bmm(output_grad);

  if (gradients_.empty()) {
    gradients_[inputs_[0]] = g0;
    gradients_[inputs_[1]] = g1;
  } else {
    gradients_[inputs_[0]].add_(g0);
    gradients_[inputs_[1]].add_(g1);
  }
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
  if (shape.size() != 2) {
    throw ValueError("Incompatible shape for softmax: " + vstr(shape.v()));
  }

  auto m = value_.reduce_max(1, true);
  value_.sub_(m).exp_();
  value_.mul_(value_.reduce_sum(1, true).recip_());

  size_t nb = shape[0]; // number of batches
  size_t js = shape[-1]; // size of the Jacobian

  derivative_.zeros({nb, js, js});

  for (size_t b = 0; b < nb; ++b) { // batch
    for (size_t i = 0; i < js; ++i) { // Jacobian row
      for (size_t j = 0; j < js; ++j) { // Jacobian col
        if (i == j) {
          float v_i = value_.get({b, i});
          float val = v_i * (1.0f - v_i);
          derivative_.set({b, i, i}, val);
        } else {
          float v_i = value_.get({b, i});
          float v_j = value_.get({b, j});
          derivative_.set({b, i, j}, -v_i * v_j);
        }
      }
    }
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

  if (x.shape().size() != 2) {
    std::cout << "x.shape.size " << x.shape().size() << std::endl;
    throw ValueError("Incompatible shape for softmax: " + vstr(x.shape()));
  }

  auto max_x = x.reduce_max(1, true);
  auto x_sub_max_x = x.sub(max_x);
  auto logsum = x_sub_max_x.exp().reduce_sum(1, true).log_();
  logsum.add_(max_x);
  auto log_sm = x.sub(logsum);
  
  // calculate derivative
  derivative_ = log_sm.exp_().sub_(y).divs_(shape[0]);
  
  // calculate CE loss
  value_ = x_sub_max_x.muls_(-1.0f).mul_(y).reduce_sum();
  value_.divs_(shape[0]);
}

void SoftmaxCrossEntropyKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(inputs_[0]->get_value().shape());
  }
  gradients_[inputs_[0]].add_(derivative_.mul(output_grad));
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
  //ln(1.0 + e^x)
  //auto ones = NDArray();
  //ones.ones(inputs_[0]->get_value().shape());
  
  //value_ = inputs_[0]->get_value().exp();
  //value_.add_(ones).log_();
  // 1.0 / (1.0 + e^-x)
  //derivative_ = inputs_[0]->get_value();
  //derivative_.muls_(-1.0f).exp_().add_(ones).recip_();

  value_ = inputs_[0]->get_value().max_filter(0.0f);
  derivative_ = value_.minimum(0.0f, 1.0f);
}

void ReLUKernel::backward(const NDArray& output_grad) {
  if (gradients_.empty()) {
    gradients_[inputs_[0]].zeros(derivative_.shape());
  }

  gradients_[inputs_[0]].add_(output_grad.mul(derivative_));
}


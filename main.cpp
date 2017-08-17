#include <iostream>
#include <iomanip>
#include "graph.h"
#include "kernel.h"
#include "ndarray.h"
#include "mnist.h"


OpRef Linear(OpRef x, size_t inp_size, size_t out_size) {
  auto W = Variable::create(x->graph(), Shape({inp_size, out_size}), true);
  auto b = Variable::create(x->graph(), {out_size}, true);
  
  // init
  float stddev = std::sqrt(6.0f / (inp_size + out_size));
  W->set_value(NDArray({inp_size, out_size},
        random_normal_vec<float>(inp_size * out_size, 0.0f, stddev)));

  return x->bmm(W)->add(b);
}

void sgd(GraphRef g, float learning_rate) {
  auto variables = g->get_variables();

  for (auto& var : variables) {
    auto v = var->get_value();
    const auto& batch_grad = g->gradient(var);
    //std::cout << "sum(grad): " << batch_grad.reduce_sum() << std::endl;
    v.sub_(batch_grad.muls(learning_rate));
    var->set_value(v);
  }
}

NDArray one_hot(size_t batch_size, size_t classes, const std::vector<int>& data) {
  NDArray result({batch_size, classes});
  for (size_t i = 0; i < data.size(); ++i) {
    result.set({i, size_t(data[i])}, 1.0f);
  }
  result.reshape({batch_size, classes});
  return result;
}

void print_stat(const std::vector<int>& labels, const NDArray& predictions) {
  auto prediction_labels = predictions.argmax(1).vec();
  int match = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] == prediction_labels[i]) {
      match++;
    }
    // std::cout << "(" << labels[i] << ", " << int(prediction_labels[i]) << ") ";
  }
  // std::cout << std::endl;
  float accuracy = float(match) / labels.size(); 
  std::cout << "acc: " << accuracy << std::endl;
}

// #include <xmmintrin.h>

int main() {
  // _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
  MNIST mnist;
  mnist.load("mnist", true);

  GraphRef g = std::make_shared<Graph>();
  
  auto X = Variable::create(g, {28 * 28});
  auto y = Variable::create(g, {10});

  auto l1 = Linear(X, 28*28, 512)->relu();
  auto l2 = Linear(l1, 512, 512)->relu();
  auto l3 = Linear(l1, 512, 10);
  auto loss = l3->softmax_ce(y);
  auto pred = l3->softmax();
  
  size_t batch_size = 100;
  size_t classes = 10;
  int steps = 1000;
  float epoch = 0;

  for (int i = 0; i < steps; ++i) {
    auto batch = mnist.get_train_batch(batch_size);
    auto& batch_X = std::get<0>(batch);
    auto& batch_y = std::get<1>(batch);

    X->set_value(NDArray({batch_size, 28*28}, batch_X));
    y->set_value(one_hot(batch_size, classes, batch_y));
    g->forward();

    epoch += float(batch_size) / float(mnist.train_size);
    std::cout << std::fixed << std::setw(6) << std::setprecision(6)
      << "epoch: " << epoch << "\tloss: " << loss->get_value() <<std::endl;
    print_stat(batch_y, pred->get_value());
    g->backward(loss);
    sgd(g, 0.1);
  }
  //std::cout << pred->get_value() << std::endl;
  return 0;
}


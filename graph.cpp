#include <queue>
#include <string>

#include "util.h"
#include "graph.h"
#include "kernel.h"

Node::~Node() {}

NodeRef Node::ref() {
  return shared_from_this();
}

GraphRef Node::graph() {
  return graph_;
}

const NDArray& Node::get_value() const {
  return kernel()->get_value();
}

std::string Node::str() const {
  return "node";
}

Node::Node(GraphRef graph) 
  : graph_(graph) { }


Op::Op(const Op::protected_&, GraphRef graph) 
  : Node(graph) { }

Op::~Op() {}

OpRef Op::ref() {
  return std::static_pointer_cast<Op>(shared_from_this());
}

Op::operator NodeRef () {
  return Node::ref();
}

template <class K, class... Args>
OpRef Op::op(const std::string& name, Args... args) {
  std::vector<NodeRef> argv = {Node::ref(), args...};
  
  for (size_t i = 1; i < argv.size(); ++i) {
    if (argv[0]->graph() != argv[i]->graph()) {
      throw RuntimeError(name + ": trying to extend grapth with node from a different graph");
    }
  }

  auto kernel = std::make_shared<K>();
  kernel->set_inputs(argv);

  auto op = std::make_shared<Op>(protected_{0}, argv[0]->graph()); 
  op->set_kernel(kernel);
  op->graph()->add(std::static_pointer_cast<Node>(op));
  return op;
}

template OpRef Op::op<AddKernel, NodeRef>(const std::string&, NodeRef);
template OpRef Op::op<SubKernel, NodeRef>(const std::string&, NodeRef);
template OpRef Op::op<MulKernel, NodeRef>(const std::string&, NodeRef);
template OpRef Op::op<DotKernel, NodeRef>(const std::string&, NodeRef);
template OpRef Op::op<BatchMatMulKernel, NodeRef>(const std::string&, NodeRef);
template OpRef Op::op<SoftmaxKernel>(const std::string&);
template OpRef Op::op<SoftmaxCrossEntropyKernel>(const std::string&, NodeRef);
 

OpRef Op::add(NodeRef other) {
  return op<AddKernel, NodeRef>("add", other);
}

OpRef Op::sub(NodeRef other) {
  return op<SubKernel, NodeRef>("sub", other);
}

OpRef Op::mul(NodeRef other) {
  return op<MulKernel, NodeRef>("mul", other);
}

OpRef Op::dot(NodeRef other) {
  return op<DotKernel, NodeRef>("dot", other);
}

OpRef Op::bmm(NodeRef other) {
  return op<BatchMatMulKernel, NodeRef>("bmm", other);
}

OpRef Op::softmax() {
  return op<SoftmaxKernel>("softmax");
}

OpRef Op::softmax_ce(NodeRef other) {
  return op<SoftmaxCrossEntropyKernel>("softmax ce", other);
}

OpRef Op::relu() {
  return op<ReLUKernel>("relu");
}

std::string Op::str() const {
  return "op: {\n"
    + kernel_->str()
    + " = "
    + kernel_->get_value().str()
    + "\n}";
}


VariableRef Variable::create(GraphRef graph,
    const Shape& shape, bool requires_grad/* = true */) {
  return std::make_shared<Variable>(Op::protected_{0}, graph, shape, requires_grad); 
}

Variable::Variable(const Op::protected_& p, GraphRef graph,
    const Shape& shape, bool requires_grad)
  : Op(p, graph)
  , kernel_(std::make_shared<ValueKernel>(shape))
  , shape_(shape)
  , requires_grad_(requires_grad) { }

Variable::~Variable() { }

VariableRef Variable::ref() {
  return std::static_pointer_cast<Variable>(shared_from_this());
}

Variable::operator NodeRef() {
  return Node::ref();
}

Variable::operator OpRef() {
  return Op::ref();
}

const Shape& Variable::shape() const {
  return shape_;
}

void Variable::set_value(const NDArray& value) {
  const auto& shape1 = shape_;
  const auto& shape2 = value.shape();

  if (shape1 != shape2) {
    int s1 = shape1.size();
    int s2 = shape2.size();
    bool ok = false;
    
    if (std::abs(s1 - s2) <= 1) {
      ok = true;
      for (size_t i = 1; i <= std::min(s1, s2); ++i) {
        ok = (shape1[s1 - i] == shape2[s2 - i]);
        if (!ok) break;
      }
    }

    if (!ok) {
      // TODO
      throw IncompatibleShapes("set_value", {shape1.v(), shape2});
    }
  }

  kernel_->set_value(value);
}

std::string Variable::str() const {
  return "var: {\n" + kernel_->str() + "\n}";
}

KernelRef Variable::kernel() const {
  return std::static_pointer_cast<Kernel>(kernel_);
}

bool Variable::requires_grad() const {
  return requires_grad_;
}


void Graph::add(NodeRef node) {
  auto& inputs = node->kernel()->get_inputs();
  for (auto& input_node : inputs) {
    adj_[input_node].push_back(node);
  }
}

void Graph::forward() {
  std::unordered_map<NodeRef, int> input_cnt;

  for (const auto& item : adj_) {
    const auto& node = item.first;
    input_cnt[node] = 0;
  }

  for (const auto& item : adj_) {
    const auto& suc_u = item.second;

    for (const auto& v : suc_u) {
      input_cnt[v]++;
    }
  }

  std::queue<NodeRef> q;

  for (const auto& item : input_cnt) {
    const auto& node = item.first;
    int node_in_cnt = item.second;

    if (node_in_cnt == 0) {
      q.push(node);
    }
  }

  top_order_.clear();
  int count = 0;

  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    top_order_.push_back(node);

    for (const auto& u : adj_[node]) {
      if (--input_cnt[u] == 0) {
        q.push(u);
      }
    }

    count++;
  }

  if (count != adj_.size()) {
    throw RuntimeError("graph contains cycle");
  }

  for (const auto& node : top_order_) {
    node->kernel()->forward();
  }
}
 

void Graph::backward(NodeRef node) {
  gradients_.clear();
 
  size_t i = top_order_.size();
  for (; i != 0; --i) {
    if (top_order_[i-1] == node) break;
  }

  if (i == 0) {
    throw RuntimeError("cannot backprop from unknown node");
  }

  for (; i != 0; --i) {
    auto curr_node = top_order_[i-1];
    auto kernel = curr_node->kernel();
    bool leaf_node = kernel->get_inputs().empty();

    kernel->clear_gradients();

    if (adj_[curr_node].empty()) {
      auto output_grad = NDArray({1}, {1});
      if (!leaf_node) {
        kernel->backward(output_grad);
      } else {
        gradients_[curr_node] = output_grad;
      }
    } else {
      for (auto& output_node : adj_[curr_node]) {
        auto output_grad = output_node->kernel()->get_gradient(curr_node);
        
        if (!leaf_node) {
          kernel->backward(output_grad);
        } else {
          if (gradients_.count(curr_node) == 0) {
            gradients_[curr_node] = output_grad;
          } else {
            gradients_[curr_node].add_(output_grad);
          }
        }
      }
    }
  }
}

NDArray Graph::gradient(const NodeRef& node) const {
  auto it = gradients_.find(node);
  if (it != gradients_.end()) {
    return it->second;
  }
 
  return NDArray();
}

std::ostream& operator<<(std::ostream& os, const NodeRef& node) {
  os << node->str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const OpRef& node) {
  os << node->str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const VariableRef& node) {
  os << node->str();
  return os;
}


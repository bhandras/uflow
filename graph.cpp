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
template OpRef Op::op<MatMulKernel, NodeRef>(const std::string&, NodeRef);
template OpRef Op::op<SoftmaxKernel>(const std::string&);
 

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

OpRef Op::mm(NodeRef other) {
  return op<MatMulKernel, NodeRef>("mm", other);
}

OpRef Op::softmax() {
  return op<SoftmaxKernel>("softmax");
}

std::string Op::str() const {
  return "op: {\n"
    + kernel_->str()
    + " = "
    + kernel_->get_value().str()
    + "\n}";
}

Op::Op(const Op::protected_&, GraphRef graph) 
  : Node(graph) { }


VariableRef Variable::create(GraphRef graph, bool requires_grad/* = true */) {
  return std::make_shared<Variable>(Op::protected_{0}, graph, requires_grad); 
}

Variable::~Variable() { }

VariableRef Variable::ref() {
  return std::static_pointer_cast<Variable>(shared_from_this());
}

Variable::operator NodeRef() {
  return Node::ref();
}

void Variable::set_value(const NDArray& value) {
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

Variable::Variable(const Op::protected_& p, GraphRef graph, bool requires_grad)
  : Op(p, graph)
  , kernel_(std::make_shared<ValueKernel>())
  , requires_grad_(requires_grad) { }


void Graph::add(NodeRef node) {
  auto& inputs = node->kernel()->get_inputs();
  for (auto& input_node : inputs) {
    adj_[input_node].push_back(node);
  }
}

void Graph::eval() {
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

  std::vector<NodeRef> top_order;
  int count = 0;

  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    top_order.push_back(node);

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

  for (const auto& node : top_order) {
    node->kernel()->forward();
  }
  
  gradients_.clear();
  
  for (size_t i = top_order.size(); i != 0; --i) {
    auto node = top_order[i - 1];
    auto kernel = node->kernel();
    bool leaf_node = kernel->get_inputs().empty();

    if (adj_[node].empty()) {
      auto output_grad = NDArray({1}, {1});
      if (!leaf_node) {
        kernel->backward(output_grad);
      } else {
        gradients_[node] = output_grad;
      }
    } else {
      for (auto& output_node : adj_[node]) {
        auto output_grad = output_node->kernel()->get_gradient(node);
        
        if (!leaf_node) {
          kernel->backward(output_grad);
        } else {
          if (gradients_.count(node) == 0) {
            gradients_[node] = output_grad;
          } else {
            gradients_[node].add_(output_grad);
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


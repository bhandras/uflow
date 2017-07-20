#include <string>
#include <iostream>

#include "util.h"
#include "graph.h"
#include "kernel.h"

Node::Node(std::unique_ptr<Kernel> kernel)
  : kernel_(std::move(kernel)) { }


void Node::eval() {
  kernel_->forward();
}

const NDArray& Node::value() const {
  return kernel_->value();
}

std::string Node::str() const {
  return kernel_->str();
}

Node::ptr Graph::var(NDArray value) {
  auto var_node = std::make_shared<Node>(std::make_unique<Value>(value));
  return var_node;
}

Node::ptr Graph::add(Node::ptr a, Node::ptr b) {
  auto add_node = std::make_shared<Node>(std::make_unique<Add>(a, b));
  adj_[a].push_back(add_node);
  adj_[b].push_back(add_node);

  return add_node;
}

Node::ptr Graph::mul(Node::ptr a, Node::ptr b) {
  auto mul_node = std::make_shared<Node>(std::make_unique<Mul>(a, b));
  adj_[a].push_back(mul_node);
  adj_[b].push_back(mul_node);

  return mul_node;
}

Node::ptr Graph::dot(Node::ptr a, Node::ptr b) {
  auto dot_node = std::make_shared<Node>(std::make_unique<Dot>(a, b));
  adj_[a].push_back(dot_node);
  adj_[b].push_back(dot_node);

  return dot_node;
}

Node::ptr Graph::mm(Node::ptr a, Node::ptr b) {
  auto mm_node = std::make_shared<Node>(std::make_unique<MatMul>(a, b));
  adj_[a].push_back(mm_node);
  adj_[b].push_back(mm_node);

  return mm_node;
}

void Graph::eval() {
  std::unordered_map<Node::ptr, std::list<Node::ptr>> pre;
  std::unordered_map<Node::ptr, int> input_cnt;

  for (const auto& item : adj_) {
    const auto& node = item.first;
    input_cnt[node] = 0;
  }

  for (const auto& item : adj_) {
    const auto& u = item.first;
    const auto& suc_u = item.second;

    for (const auto& v : suc_u) {
      input_cnt[v]++;
      pre[v].push_front(u);
    }
  }

  std::queue<Node::ptr> q;

  for (const auto& item : input_cnt) {
    const auto& node = item.first;
    int node_in_cnt = item.second;

    if (node_in_cnt == 0) {
      q.push(node);
    }
  }

  std::vector<Node::ptr> top_order;
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
    node->kernel().forward();
  }

  std::cout << "eval res: " <<top_order[top_order.size() - 1]->value() << std::endl;

  std::cout << std::endl << "backprop" << std::endl;

  std::unordered_map<Node::ptr, std::unordered_map<Node::ptr, NDArray>> work_gradients;
  gradients_.clear();
  auto dummy = std::list<Node::ptr>{var(NDArray({1}, {1}))};

  for (int i = top_order.size() - 1; i >= 0; --i) {
    auto& node = top_order[i];
    auto& suc_nodes = adj_[top_order[i]].empty() ? dummy : adj_[top_order[i]];
    auto& pre_nodes = pre[top_order[i]];
    std::cout << "i: " << i << " - " << (top_order.size() - 1) << std::endl;
    node->kernel().backward(suc_nodes, work_gradients[node]);

    if (pre_nodes.empty()) {
      for (auto& n_suc : suc_nodes) {
        const auto& grad_suc = work_gradients[n_suc][node];

        // leaf node
        if (gradients_.count(node) == 0) {
          gradients_[node] = grad_suc;
        } else {
          gradients_[node].add_(grad_suc);
        }
      }      
    }
  }
}

NDArray Graph::gradient(const Node::ptr& node) const {
  auto it = gradients_.find(node);
  if (it != gradients_.end()) {
    return it->second;
  }
  
  return NDArray();
}


std::ostream& operator<<(std::ostream& os, const Node& node) {
  os << node.str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const Node::ptr& node) {
  os << node->str();
  return os;
}


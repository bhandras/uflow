#include <iostream>

#include "graph.h"
#include "kernel.h"

Node::Node(std::unique_ptr<Kernel> kernel)
  : kernel_(std::move(kernel)) { }


void Node::eval() {
  kernel_->forward();
}

float Node::value() const {
  return kernel_->value();
}

float Node::gradient(const Node::ptr& node) {
  return kernel_->gradient(node);
}

Node::ptr Graph::var(float value) {
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
    std::cout << "circle" << std::endl;
  }


  for (const auto& node : top_order) {  
    node->eval();
  }

  gradients_.clear();
  gradients_[top_order.back()] = 1.0f;

  for (int i = top_order.size() - 1; i >= 0; --i) {
    std::cout << "node: " << top_order[i]->value() << std::endl;
    for (auto& n_suc : adj_[top_order[i]]) {
      float grad_suc = gradients_[n_suc];
      std::cout << "grad_suc: " << n_suc->value() << ", "<< grad_suc << std::endl;

      for (auto& n_pre : pre[top_order[i]]) {
        std::cout << "grad_pre: " << n_pre->value() << "; "<< top_order[i]->gradient(n_pre) << std::endl;
        gradients_[n_pre] += grad_suc * top_order[i]->gradient(n_pre);
      }
    }
  }
}

float Graph::gradient(const Node::ptr& node) const {
  auto it = gradients_.find(node);
  if (it != gradients_.end()) {
    return it->second;
  }

  return 0.0f;
}


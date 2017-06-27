#include <string>
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

std::string Node::to_string() const {
  return kernel_->to_string();
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

  std::unordered_map<Node::ptr, std::unordered_map<Node::ptr, float>> work_gradients;
  gradients_.clear();
  
  for (int i = top_order.size() - 1; i >= 0; --i) {
    auto& node = top_order[i];
    auto& suc_nodes = adj_[top_order[i]];
    auto& pre_nodes = pre[top_order[i]];

    std::cout << "node: " << node->to_string() << std::endl;
    
    if (suc_nodes.empty()) {
      // result (last) node
      float grad_suc = 1.0f;

      for (auto& p : pre_nodes) {
        std::cout << "grad_pre: " << p->to_string() << "; " << node->gradient(p) << std::endl;

        work_gradients[node][p] += grad_suc * node->gradient(p);
      }
    } else { 
      for (auto& s : suc_nodes) {
        float grad_suc = work_gradients[s][node];
        
        std::cout << "grad_suc: " << s->to_string() << ", " << grad_suc << std::endl;

        if (!pre_nodes.empty()) {
          for (auto& n_pre : pre_nodes) {
            
            std::cout << "grad_pre: " << n_pre->to_string() << "; " 
              << node->gradient(n_pre) << std::endl;
            
            work_gradients[node][n_pre] += grad_suc * node->gradient(n_pre);
          }
        } else {
          // leaf node
          gradients_[top_order[i]] += grad_suc;
        }
      }
    }
    
    std::cout << "---" << std::endl;
  }
}

float Graph::gradient(const Node::ptr& node) const {
  auto it = gradients_.find(node);
  if (it != gradients_.end()) {
    return it->second;
  }

  return 0.0f;
}


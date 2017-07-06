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

NDArray Node::gradient(const Node::ptr& node) {
  return kernel_->gradient(node);
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
    node->eval();
  }

  std::unordered_map<Node::ptr, std::unordered_map<Node::ptr, NDArray>> work_gradients;
  gradients_.clear();
  
  for (int i = top_order.size() - 1; i >= 0; --i) {
    auto& node = top_order[i];
    auto& suc_nodes = adj_[top_order[i]];
    auto& pre_nodes = pre[top_order[i]];

    
    if (suc_nodes.empty()) {
      std::cout << "result node: " << node << std::endl;
      // result (last) node => grad_suc = 1
      for (auto& n_pre : pre_nodes) {

        // grad of 'node' w.r.t 'n_pre'
        auto grad_pre = node->gradient(n_pre);
        std::cout << "grad_pre: " << n_pre << "; " << grad_pre << std::endl;

        if (work_gradients[node].count(n_pre) == 0) {
          work_gradients[node][n_pre] = grad_pre;
        } else {
          work_gradients[node][n_pre].add_(grad_pre);
        }
      }
    } else {
      std::cout << "node: " << node << std::endl;
      for (auto& n_suc : suc_nodes) {
        const auto& grad_suc = work_gradients[n_suc][node];
        
        std::cout << "grad_suc: " << n_suc << "; " << grad_suc << std::endl;

        if (!pre_nodes.empty()) {
          for (auto& n_pre : pre_nodes) {
           
            // grad of 'node' w.r.t 'n_pre'
            auto grad_pre = node->gradient(n_pre);
            std::cout << "grad_pre: " << n_pre << "; " << grad_pre << std::endl; 
            auto grad = grad_suc.mul(grad_pre);

            if (work_gradients[node].count(n_pre) == 0) {
              work_gradients[node][n_pre] = grad;
            } else {
              work_gradients[node][n_pre].add_(grad);
            }
          }
        } else {
          // leaf node
          if (gradients_.count(node) == 0) {
            gradients_[node] = grad_suc;
          } else {
            gradients_[node].add_(grad_suc);
          }
        }
      }
    }
    
    std::cout << "---" << std::endl;
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


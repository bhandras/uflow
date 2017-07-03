#ifndef _ndarray_h_
#define _ndarray_h_

#include <stack>
#include <vector>
#include <sstream>

class NDArray {
  public:
    NDArray(const std::vector<size_t>& shape)
      : shape_(shape) {
        update_shape();
        arr_ = std::vector<float>(size_, 0);
        for (int i = 0; i < size_; ++i) {
          arr_[i] = i;
        }
    }

    void update_shape() {
      size_ = 1;
      for (auto dim : shape_) {
        size_ *= dim;
      }

      stride_ = std::vector<size_t>(shape_.size(), 1);
      for (int i = shape_.size() - 2; i >= 0; --i) {
        stride_[i] = shape_[i + 1] * stride_[i + 1];
      }
    }

    const std::vector<size_t>& shape() const {
      return shape_;
    }

    bool reshape(const std::vector<size_t>& shape) {
      size_t new_size = 1;
      for (auto dim : shape) {
        new_size *= dim;
      }

      if (new_size != size_) {
        return false;
      }

      shape_ = shape;
      update_shape();
      return true;
    }

    void to_string_helper(std::stringstream& ss, int dim, int& pos, int level) const {
      int ndim = shape_.size();
      if (dim == ndim - 1) {
        ss << "[";
        for (int i = 0; i < shape_[dim]; ++i) {
          ss << arr_[pos++];
          if (i != shape_[dim] - 1) {
            ss << ", ";
          }
        }
        ss << "]";
      } else {
        ss << "[";
        for (int i = 0; i < shape_[dim]; ++i) {
          to_string_helper(ss, dim + 1, pos, level + 1);
          if (i != shape_[dim] - 1) {
            ss << "," << std::endl << std::string(level, ' ');
          }
        }
        ss << "]";
      }
    }

    std::string to_string() const {
      std::stringstream ss;
      int pos = 0;
      to_string_helper(ss, 0, pos, 1);
      return ss.str();
    }

  private:
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    size_t size_;
    std::vector<float> arr_;
};


#endif // _ndarray_h_


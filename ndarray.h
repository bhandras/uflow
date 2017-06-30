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

    void print1(std::stringstream& ss, int& pos, int padding) const {
      int n = shape_[shape_.size() - 1];

      if (padding > 0) {
        ss << std::string(padding, ' ');
      }
      ss << "[ ";
      for (int i = 0; i < n; ++i) {
        ss << arr_[pos++];
        if (i != n - 1) {
          ss << ", ";
        }
      }
      ss << " ]";
    }

    void print2(std::stringstream& ss, int& pos, int padding) const {
      int ndim = shape_.size();
      int n = shape_[ndim - 2];

      if (padding > 0) {
        ss << std::string(padding, ' ');
      }
 
      ss << "[";
      for (int i = 0; i < n; ++i) {
        print1(ss, pos, i == 0 ? 0 : padding+1);
        if (i != n - 1) {
          ss << ", " << std::endl;
        }
      }

      ss << "]";
    }

    void print3(std::stringstream& ss, int& pos, int padding) const {
      int ndim = shape_.size();
      int n = shape_[ndim - 3];

      if (padding > 0) {
        ss << std::string(padding, ' ');
      }
      
      ss << "[";
      for (int i = 0; i < n; ++i) {
        print2(ss, pos, i == 0 ? 0 : padding );
        if (i != n - 1) {
          ss << ", " << std::endl << std::endl;
        }
      }

      ss << "]";
   
    }

    std::string to_string() const {
      int ndim = shape_.size();
      std::stringstream ss;
      
      int pos = 0;
      if (ndim == 1) {
        print1(ss, pos, 0);
      }
      else if (ndim == 2) {
        print2(ss, pos, 0);
      }
      else if (ndim == 3) {
        print3(ss, pos, 0);
      }

      return ss.str();
    }

  private:
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    size_t size_;
    std::vector<float> arr_;
};


#endif // _ndarray_h_


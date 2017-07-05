#ifndef _ndarray_h_
#define _ndarray_h_

#include <vector>
#include "util.h"
#include "exception.h"

class NDArray {
  public:
    NDArray() 
      : size_(0) { }

    NDArray(const std::vector<size_t>& shape)
    : shape_(shape) {
      update_shape();
    }

    void ones(size_t n) {
      shape_ = std::vector<size_t>{n};
      update_shape();

      for (size_t i = 0; i < size_; ++i) {
        arr_[i] = 1.0f;
      }
    }

    void arange(size_t n, float step = 1.0f) {
      shape_ = std::vector<size_t>{n};
      update_shape();

      for (size_t i = 0; i < size_; ++i) {
        arr_[i] = i * step;
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
      arr_.resize(size_, 0.0f); 
    }

    const std::vector<size_t>& shape() const {
      return shape_;
    }

    void reshape(const std::vector<size_t>& shape) {
      size_t new_size = 1;
      for (auto dim : shape) {
        new_size *= dim;
      }

      if (new_size != size_) {
        throw NDArray::ex_incompatible_shapes("reshape", shape_, shape);
      }

      shape_ = shape;
      update_shape();
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

    NDArray& operator=(const NDArray& other) = default;

    NDArray operator+(const NDArray& other) const {
      NDArray res = *this;
      res += other;
      return res;
    }

    NDArray& operator+=(const NDArray& other) {
      if (shape_ == other.shape_) {
        for (size_t i = 0; i < arr_.size(); ++i) {
          arr_[i] += other.arr_[i];
        }
      } else {
        throw NDArray::ex_incompatible_shapes("add", shape_, other.shape_);
      }

      return *this;
    }

    NDArray operator*(const NDArray& other) const {
      NDArray res = *this;
      res *= other;
      return res;
    }

    NDArray operator*=(const NDArray& other) {
      if (shape_ == other.shape_) {
        for (size_t i = 0; i < arr_.size(); ++i) {
          arr_[i] *= other.arr_[i];
        }
      } else {
        throw NDArray::ex_incompatible_shapes("mul", shape_, other.shape_);
      }

      return *this;
    }

    NDArray dot(const NDArray& other) const {
      if (shape_.size() != other.shape_.size()) {
        throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
      }

      size_t ndim = shape_.size();
      if (ndim > 1 && shape_[ndim - 1] != other.shape_[ndim - 2]) {
        throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
      }

      if (ndim > 2) {
        for (size_t i = 0; i < ndim - 2; ++i) {
          if (shape_[i] != other.shape_[i]) {
            throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
          }
        }
      }

      if (ndim == 1) {
        NDArray res({1});

        for (size_t i = 0; i < ndim; ++i) {
          res.arr_[0] += arr_[i] * other.arr_[i];
        }

        return res;
      } 
      
      // matrix mul
      std::vector<size_t> res_shape(shape_);
      res_shape[ndim - 1] = other.shape_[ndim - 1]; 
      NDArray res(res_shape);

      size_t count = 1;
      for (size_t i = 0; i < ndim - 2; ++i) {
        count *= shape_[i];
      }

      size_t m = shape_[ndim - 2];
      size_t n = shape_[ndim - 1];
      size_t k = other.shape_[ndim - 1];
      size_t stride = m * k;

      for (int c = 0; c < count; ++c) {
        size_t offset = c * stride;
        auto A = &arr_[offset];
        auto B = &other.arr_[offset];
        auto AB = &res.arr_[offset];

        for (size_t i = 0; i < m; ++i) {
          for (size_t j = 0; j < k; ++j) {
            
            auto& AB_ij = AB[i * k + j];
            auto A_i = &A[i * n];
            auto B_j = &B[j];
            
            for (size_t l = 0; l < n; ++l) {
              AB_ij += A_i[l] * B_j[l * k];
            }
          }
        }
      }

      return res;
    }

    static ValueError ex_incompatible_shapes(
        const char* prefix,
        const std::vector<size_t>& a,
        const std::vector<size_t>& b) {
       
      return ValueError(std::string(prefix)
          + ": "
          + "incompatible shapes "
          + str(a)
          + " and "
          + str(b));
    }

  private:
    size_t size_;
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    std::vector<float> arr_;
};


#endif // _ndarray_h_


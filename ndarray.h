#ifndef _ndarray_h_
#define _ndarray_h_

#include <vector>
#include "util.h"
#include "exception.h"


class NDArray {
  public:
    NDArray() 
      : size_(0) { }

    NDArray(const std::vector<size_t>& shape,
        const std::vector<float>& init=std::vector<float>())
    : shape_(shape) {
      update_shape();
      if (!init.empty()) {
        for (size_t i = 0; i < size_; ++i) {
          arr_[i] = init[i % init.size()];  
        }
      }
    }

    void unsqueeze(size_t dim) {
      if ((shape_.empty() && dim != 0)
          || dim > shape_.size()) {
        throw RuntimeError("cannot unsqueeze "
            + vstr(shape_)
            + " at "
            + std::to_string(dim));
      }

      shape_.insert(shape_.begin() + dim, 1);
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

    void str_helper(std::stringstream& ss, int dim, int& pos, int level) const {
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
          str_helper(ss, dim + 1, pos, level + 1);
          if (i != shape_[dim] - 1) {
            ss << "," << std::endl << std::string(level, ' ');
          }
        }
        ss << "]";
      }
    }

    std::string str() const {
      if (shape_.empty()) {
        return "[]";
      }
      std::stringstream ss;
      int pos = 0;
      str_helper(ss, 0, pos, 1);
      return ss.str();
    }

    NDArray& operator=(const NDArray& other) = default;
    
    NDArray add(const NDArray& other) const {
      NDArray tmp = *this;
      return tmp.add_(other);
    }

    NDArray& add_(const NDArray& other) {
      if (shape_ != other.shape_) {
        throw NDArray::ex_incompatible_shapes("add", shape_, other.shape_);
      }

      for (size_t i = 0; i < arr_.size(); ++i) {
        arr_[i] += other.arr_[i];
      }

      return *this;
    }

    NDArray mul(const NDArray& other) const {
      NDArray tmp = *this;
      return tmp.mul_(other);
    }

    NDArray& mul_(const NDArray& other) {
      if (shape_ != other.shape_) {
        throw NDArray::ex_incompatible_shapes("add", shape_, other.shape_);
      }

      for (size_t i = 0; i < arr_.size(); ++i) {
        arr_[i] *= other.arr_[i];
      }

      return *this;
    }

    NDArray dot(const NDArray& other) const {
      if (shape_.size() != other.shape_.size()) {
        throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
      }

      size_t ndim = shape_.size();
      if (ndim >= 1 && shape_[ndim - 1] != other.shape_[ndim - 1]) {
        throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
      }

      if (ndim > 1) {
        for (size_t i = 0; i < ndim - 1; ++i) {
          if (shape_[i] != other.shape_[i]) {
            throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
          }
        }
      }

      std::vector<size_t> res_shape(shape_);
      res_shape[ndim - 1] = 1; 
      NDArray res(res_shape);

      size_t count = 1;
      for (size_t i = 0; i < ndim - 1; ++i) {
        count *= shape_[i];
      }

      size_t stride = shape_[ndim - 1];
      for (int c = 0; c < count; ++c) {
        size_t offset = c * stride;
        auto& ai_dot_bi = res.arr_[offset];
        auto ai = &res.arr_[offset];
        auto bi = &other.arr_[offset];

        for (size_t j = 0; j < shape_[ndim - 1]; ++j) {
          ai_dot_bi += ai[j] * bi[j];
        }
      } 
      
      return res;
    }

    NDArray mm(const NDArray& other) const {
      if (shape_.size() != other.shape_.size()) {
        throw NDArray::ex_incompatible_shapes("mm", shape_, other.shape_);
      }

      size_t ndim = shape_.size();
      if (ndim >= 2 && shape_[ndim - 1] != other.shape_[ndim - 2]) {
        throw NDArray::ex_incompatible_shapes("mm", shape_, other.shape_);
      }

      if (ndim > 2) {
        for (size_t i = 0; i < ndim - 2; ++i) {
          if (shape_[i] != other.shape_[i]) {
            throw NDArray::ex_incompatible_shapes("mm", shape_, other.shape_);
          }
        }
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
          + vstr(a)
          + " and "
          + vstr(b));
    }

  private:
    size_t size_;
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    std::vector<float> arr_;
};

std::ostream& operator<<(std::ostream& os, const NDArray& arr);

#endif // _ndarray_h_


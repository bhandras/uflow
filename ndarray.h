#ifndef _ndarray_h_
#define _ndarray_h_

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include "util.h"
#include "exception.h"

class NDArray;
std::ostream& operator<<(std::ostream& os, const NDArray& arr);

class NDArray {
  public:
    NDArray() {} 

    NDArray(const std::vector<size_t>& shape,
            const std::vector<float>& init = std::vector<float>())
      : shape_(shape) {
      update_shape();
      
      if (!init.empty()) {
        for (size_t i = 0; i < arr_.size(); ++i) {
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

      for (auto& val : arr_) {
        val = 1.0f;
      }
    }

    void arange(size_t n, float step = 1.0f) {
      shape_ = std::vector<size_t>{n};
      update_shape();

      for (size_t i = 0; i < arr_.size(); ++i) {
        arr_[i] = i * step;
      }
    }

    void update_shape() {
      size_t size = 1;
      for (auto dim : shape_) {
        size *= dim;
      }

      arr_.resize(size, 0.0f); 
      
      strides_ = std::vector<size_t>(shape_.size(), 1);
      for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = shape_[i + 1] * strides_[i + 1];
      }
    }

    const std::vector<size_t>& shape() const {
      return shape_;
    }

    void reshape(const std::vector<size_t>& shape) {
      size_t new_size = 1;
      for (auto dim : shape) {
        new_size *= dim;
      }

      if (new_size != arr_.size()) {
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
    
    NDArray transpose() const {
      if (shape_.size() < 2) {
        throw RuntimeError("cannot transpose array");
      }

      NDArray res(shape_);

      size_t i1 = res.shape_.size() - 2;
      size_t i2 = res.shape_.size() - 1;
      size_t d1 = res.shape_[i1];
      size_t d2 = res.shape_[i2];
      
      size_t count = 1;
      for (size_t i = 0; i < shape_.size() - 2; ++i) {
        count *= shape_[i];
      }
      
      size_t stride = d1 * d2;

      for (size_t c = 0; c < count; ++c) {
        size_t offset = c * stride;
        auto oa = &arr_[offset];
        auto ra = &res.arr_[offset];
        
        for (size_t i = 0; i < d1; ++i) {
          for (size_t j = 0; j < d2; ++j) {
            ra[j * d1 + i] = oa[i * d2 + j];
          }
        }
      }

      std::swap(res.shape_[i1], res.shape_[i2]);
      return res;
    }


    std::vector<size_t> get_common_shape(const NDArray& other) const {
      auto& shape1 = shape_;
      auto& shape2 = other.shape_;

      size_t s1 = shape1.size();
      size_t s2 = shape2.size();
      size_t sc = std::max(s1, s2);
      std::vector<size_t> common_shape(sc);
      
      for (size_t i = 0; i < common_shape.size(); ++i) {
        size_t d1 = i < s1 ? shape1[s1 - i - 1] : 1;
        size_t d2 = i < s2 ? shape2[s2 - i - 1] : 1;
        if (std::min(d1, d2) != 1 && d1 != d2) {
          return std::vector<size_t>();
        }
        common_shape[sc - i - 1] = std::max(d1, d2);
      }
      
      return common_shape;
    }

    std::vector<size_t> strides(const std::vector<size_t>& shape_d) const {
      // get broadcast strides for broadcast shape
      auto& shape_s = shape_;
      auto& strides_s = strides_;
      std::vector<size_t> strides_d(shape_d.size(), 0);

      size_t offs = shape_d.size() - shape_s.size();
      for (size_t i = shape_s.size(); i > 0; --i) {
        size_t idx = i - 1;
        if (shape_d[idx + offs] == shape_s[idx]) {
          strides_d[idx + offs] = strides_s[idx];
        }
      }

      return strides_d;
    }


    NDArray expand_as(const NDArray& other) const {
      return expand(other.shape());
    }

    NDArray expand(const std::vector<size_t>& new_shape) const {
      if (shape_ == new_shape) {
        return *this;
      }

      NDArray res(new_shape);
      auto strides_d = strides(new_shape);
      
      std::vector<size_t> nonzero_strides;
      for (auto s : strides_d) {
        if (s != 0) {
          nonzero_strides.push_back(s);
        }
      }

      for (size_t pos = 0; pos < res.arr_.size(); ++pos) {
        size_t inc = 0;
        size_t tmp = pos;
        for (size_t i = new_shape.size(); i > 0; --i) {
          size_t idx = i - 1;
          size_t n = tmp % new_shape[idx];
          inc += n * strides_d[idx];
          tmp /= new_shape[idx];
        }
        if (tmp > 0) {
          inc = new_shape[0] * nonzero_strides[0];
        }
        res.arr_[pos] = arr_[inc];
      }

      return res;
    }

    NDArray add(const NDArray& other) const {
      NDArray tmp = *this;
      return tmp.add_(other);
    }

    NDArray& add_(const NDArray& other) {
      if (shape_ != other.shape_) {
        auto common_shape = get_common_shape(other);
        if (common_shape.empty()) {
          throw NDArray::ex_incompatible_shapes("add", shape_, other.shape_);
        }

        if (common_shape != shape_) {
          *this = expand(common_shape);
        }

        return add_(other.expand(common_shape));
      }

      for (size_t i = 0; i < arr_.size(); ++i) {
        arr_[i] += other.arr_[i];
      }

      return *this;
    }

    NDArray sub(const NDArray& other) const {
      NDArray tmp = *this;
      return tmp.sub_(other);
    }

    NDArray& sub_(const NDArray& other) {
      if (shape_ != other.shape_) {
        auto common_shape = get_common_shape(other);
        if (common_shape.empty()) {
          throw NDArray::ex_incompatible_shapes("sub", shape_, other.shape_);
        }

        if (common_shape != shape_) {
          *this = expand(common_shape);
        }

        return sub_(other.expand(common_shape));
      }

      for (size_t i = 0; i < arr_.size(); ++i) {
        arr_[i] -= other.arr_[i];
      }

      return *this;
    }

    NDArray mul(const NDArray& other) const {
      NDArray tmp = *this;
      return tmp.mul_(other);
    }

    NDArray& mul_(const NDArray& other) {
      if (shape_ != other.shape_) {
        auto common_shape = get_common_shape(other);
        if (common_shape.empty()) {
          throw NDArray::ex_incompatible_shapes("mul", shape_, other.shape_);
        }

        if (common_shape != shape_) {
          *this = expand(common_shape);
        }

        return mul_(other.expand(common_shape));
      }
      
      for (size_t i = 0; i < arr_.size(); ++i) {
        arr_[i] *= other.arr_[i];
      }

      return *this;
    }

    NDArray dot(const NDArray& other) const {
      if (arr_.size() == 1 || other.arr_.size() == 1) {
        return mul(other);
      }
      // shape size
      if (shape_.size() != other.shape_.size()) {
        throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
      }

      // last dim
      size_t ndim = shape_.size();
      if (ndim >= 1 && shape_[ndim - 1] != other.shape_[ndim - 1]) {
        throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
      }

      // remaining
      if (ndim > 1) {
        bool good = true;
        for (size_t i = 0; good && (i < ndim - 1); ++i) {
          if (shape_[i] != other.shape_[i]) {
            good = false;
          }
        }

        if (good && (shape_[ndim - 1] != 1 && shape_[ndim - 2] != 1)) {
          good = false;
        }

        if (!good) {
          throw NDArray::ex_incompatible_shapes("dot", shape_, other.shape_);
        }
      }

      size_t stride = 0;
      size_t res_shape_size = 0;

      if (shape_[ndim - 1] == 1 || shape_[ndim - 2] == 1) {
        stride = shape_[ndim - 1] * shape_[ndim - 2];
        res_shape_size = shape_.size() - 1;
      } else {
        stride = shape_[ndim - 1];
        res_shape_size = shape_.size();
      }

      std::vector<size_t> res_shape(res_shape_size);
      for (size_t i = 0; i < res_shape_size - 1; ++i) {
        res_shape[i] = shape_[i];
      }
      res_shape[res_shape_size - 1] = 1;
      NDArray res(res_shape);

      size_t count = 1;
      for (size_t i = 0; i < res_shape_size - 1; ++i) {
        count *= res_shape[i];
      }
      
      for (int c = 0; c < count; ++c) {
        size_t offset = c * stride;
        auto& ai_dot_bi = res.arr_[c];
        auto ai = &arr_[offset];
        auto bi = &other.arr_[offset];

        for (size_t j = 0; j < stride; ++j) {
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
    std::vector<float> arr_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
};


#endif // _ndarray_h_


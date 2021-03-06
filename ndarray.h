#ifndef _ndarray_h_
#define _ndarray_h_

#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include "util.h"
#include "exception.h"

class NDArray;
std::ostream& operator<<(std::ostream& os, const NDArray& arr);

class Shape {
  public:
    static const size_t max_size = std::numeric_limits<int>::max();

    Shape(std::initializer_list<size_t> l) {
      if (l.size() > max_size) {
        throw RuntimeError("Max shape is: "
            + std::to_string(max_size));
      }
      v_.insert(v_.end(), l.begin(), l.end());
    }

    Shape(const std::vector<size_t>& shape) {
      if (shape.size() > max_size) {
        throw RuntimeError("Max shape is: "
            + std::to_string(max_size));
      }
      v_.insert(v_.end(), shape.begin(), shape.end());
    }

    size_t size() const {
      return v_.size();
    }

    bool operator==(const Shape& other) const {
      return v_ == other.v_;
    }

    bool operator==(const std::vector<size_t>& other) const {
      return v_ == other;
    }

    bool operator !=(const Shape& other) const {
      return v_ != other.v_;
    }

    size_t operator[](int index) const {
      return v_[get_index(index)];
    }

    void unsqueeze(int axis) {
      v_.insert(v_.begin() + axis, 1);
    }

    void squeeze(int axis) {
      v_.erase(v_.begin() + axis);
    }

    void swap(int d1, int d2) {
      std::swap(v_[get_index(d1)], v_[get_index(d2)]);
    }

    bool is_row_vector() const {
      size_t s = v_.size();
      // (N)
      if (s == 1) return true;
      // ..,1,N
      if (s >= 2 && v_[s - 1] >= 1 && v_[s - 2] == 1) return true;
      
      return false; 
    }

    bool is_column_vector() const {
      size_t s = v_.size();
      // ..,N,1
      if (s >= 2 && v_[s - 1] == 1 && v_[s - 2] >= 1) return true;
      
      return false;
    }

    const std::vector<size_t> v() const {
      return v_;
    }

  private:
    int get_index(int index) const {
      if ((index > 0 && index >= v_.size()) ||
          (index < 0 && -index > v_.size())) {
        throw RuntimeError("Shape::[] index out of bounds: "
            + std::to_string(index));
      }

      int offset = index < 0 ? v_.size() : 0;
      return offset + index;
    }

    std::vector<size_t> v_;
};

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

    bool operator==(const NDArray& other) const {
      if (shape_ != other.shape_) {
        return false;
      }

      for (size_t i = 0; i < arr_.size(); ++i) {
        if (arr_[i] != other.arr_[i]) {
          return false;
        }
      }

      return true;
    }

    const std::vector<float> vec() const {
      return arr_;
    }

    void squeeze(size_t axis) {
      if (shape_.empty() ||
          axis > shape_.size() ||
          shape_[axis] != 1) {
        throw RuntimeError("cannot unsqueeze "
            + vstr(shape_)
            + " at "
            + std::to_string(axis));
      }

      shape_.erase(shape_.begin() + axis);
      update_shape();
    }

    void unsqueeze(size_t axis) {
      if ((shape_.empty() && axis != 0)
          || axis > shape_.size()) {
        throw RuntimeError("cannot unsqueeze "
            + vstr(shape_)
            + " at "
            + std::to_string(axis));
      }

      shape_.insert(shape_.begin() + axis, 1);
      update_shape();
    }

    void ones(const std::vector<size_t>& shape) {
      shape_ = shape;
      update_shape();

      for (auto& val : arr_) {
        val = 1.0f;
      }
    }

    void zeros(const std::vector<size_t>& shape) {
      shape_ = shape;
      update_shape();

      for (auto& val : arr_) {
        val = 0.0f;
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
      for (auto ax : shape) {
        if (ax != 0) {
          new_size *= ax;
        }
      }

      if (new_size != arr_.size()) {
        throw IncompatibleShapes("reshape", {shape_, shape});
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
      if (arr_.empty()) {
        return res;
      }

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

    void set(const std::vector<size_t>& index, float value) {
      if (index.size() != shape_.size()) {
        throw IncompatibleShapes("NDArray::set", {index, shape_});
      }

      for (size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] <= index[i]) {
          // todo: better exception for out of bounds
          throw RuntimeError("NDArray::set: out of bounds");
        }
      }

      size_t pos = 0;
      for (size_t i = 0; i < index.size(); ++i) {
        pos += strides_[i] * index[i];
      }

      arr_[pos] = value;
    }

    float get(const std::vector<size_t>& index) const {
      if (index.size() != shape_.size()) {
        throw IncompatibleShapes("NDArray::get", {index, shape_});
      }

      for (size_t i = 0; i < shape_.size(); ++i) {
        if (shape_[i] <= index[i]) {
          // todo: better exception for out of bounds
          throw RuntimeError("NDArray::get:"
              + vstr(index)
              + " out of bounds "
              + vstr(shape_));
        }
      }

      size_t pos = 0;
      for (size_t i = 0; i < index.size(); ++i) {
        pos += strides_[i] * index[i];
      }

     return  arr_[pos];
    }

    NDArray reduce(std::function<void(float&, const float&, size_t, size_t)> op,
        int axis, bool keep_dims, float init) const {
      if (arr_.empty()) {
        throw RuntimeError("NDArray::reduce on zero-size array");
      }

      if (axis == -1) {
        auto shape = shape_;
        
        if (keep_dims) {
          auto shape = shape_;
          for (auto& ax : shape) {
            ax = 1;
          }
        } else {
          shape.resize(1);
          shape[0] = 1;
        }

        auto res = NDArray(shape);
        res.arr_[0] = init;
        for (auto i = 0; i < arr_.size(); ++i) {
          op(res.arr_[0], arr_[i], 0, i);
        }

        return res;
      }
      
      auto shape = shape_;
      if (keep_dims) {
        shape[axis] = 1;
      } else {
        shape.erase(shape.begin() + axis);
      }
      NDArray res(shape);

      auto stride = strides_[axis];
      if (stride == 1) {
        for (size_t i = 0; i < res.arr_.size(); i++) {
          res.arr_[i] = init;
          
          size_t ax_idx = 0;
          for (size_t j = i*shape_[axis]; j < (i+1)*shape_[axis]; ++j) {
            op(res.arr_[i], arr_[j], i, ax_idx);
            ++ax_idx;
          }
        }
      } else {
        int pos = 0;
        for (size_t i = 0; i < res.arr_.size(); i++) {
          int offs = stride * (i / stride) + i;
          res.arr_[pos] = init;
          
          size_t ax_idx = 0;
          for (size_t j = 0; j < shape_[axis]; ++j) {
            op(res.arr_[pos], arr_[offs + j * stride], pos, ax_idx);
            ++ax_idx;
          }
          pos++;
        }
      }

      return res;
    }

    NDArray reduce_max(int axis = -1, bool keep_dims=false) const {
      if (arr_.empty()) {
        throw RuntimeError("NDArray::reduce_max on zero-size array");
      }
      return reduce([](float& x, const float& y, size_t i, size_t ax_idx) {
            x = std::max(x, y);
          }, axis, keep_dims, std::numeric_limits<float>::min());
    }

    NDArray reduce_sum(int axis=-1, bool keep_dims=false) const {
      if (arr_.empty()) {
        throw RuntimeError("NDArray::reduce_sum on zero-size array");
      }
      return reduce([](float& x, const float& y, size_t i, size_t ax_idx) {
          x += y;
          }, axis, keep_dims, 0.0f);
    }
   
    NDArray argmax(int axis=-1) const {
      if (arr_.empty()) {
        throw RuntimeError("NDArray::reduce_sum on zero-size array");
      }
      const float init = std::numeric_limits<float>::min();
      float max_val = init;
      size_t curr_i = 0;
      return reduce([&](float& x, const float& y, size_t i, size_t ax_idx) {
          if (curr_i != i) {
            curr_i = i;
            max_val = init;
          }

          if (y > max_val) {
            max_val = y;
            x = ax_idx;
          }

          }, axis, false, 0.0f);
    }

    NDArray max_filter(float x) const {
      if (arr_.empty()) {
        throw RuntimeError("NDArray::max_filter on zero-size array");
      }

      NDArray res(shape_);
      for (auto i = 0; i < res.arr_.size(); ++i) {
        res.arr_[i] = arr_[i] >= x ? arr_[i] : x;
      }

      return res;
    }

    NDArray& clip_(float min, float max) {
      if (arr_.empty()) {
        throw RuntimeError("NDArray::clip_ on zero-size array");
      }

      for (auto& val : arr_) {
        val = std::min(std::max(val, min), max);
      }

      return *this;
    }

    NDArray minimum(float a, float b) const {
      auto tmp = *this;
      return tmp.minimum_(a, b);
    }

    NDArray& minimum_(float a, float b) {
      if (arr_.empty()) {
        throw RuntimeError("NDArray::minimum_ on zero-size array");
      }

      for (auto& val : arr_) {
        val = val <= a ? a : b;
      }

      return *this;
    }

    NDArray exp() const {
      auto tmp = *this;
      return tmp.exp_();
    }

    NDArray& exp_() {
      for (auto& val : arr_) {
        val = std::exp(val);
      }
      return *this;
    }

    NDArray log() const {
      auto tmp = *this;
      return tmp.log_();
    }

    NDArray& log_() {
      for (auto& val : arr_) {
        val = std::log(std::max(val, std::numeric_limits<float>::min()));
      }
      return *this;
    }

    NDArray recip() const {
      auto tmp = *this;
      return tmp.recip_();
    }

    NDArray& recip_() {
      if (arr_.empty()) {
        throw RuntimeError("recip on zero-size array");
      }

      for (auto& val : arr_) {
        val = 1.0f / val;
      }

      return *this;
    }
    
    NDArray add(const NDArray& other) const {
      auto tmp = *this;
      return tmp.add_(other);
    }

    NDArray& add_(const NDArray& other) {
      if (shape_ != other.shape_) {
        auto common_shape = get_common_shape(other);
        if (common_shape.empty()) {
          throw IncompatibleShapes("NDArray::add", {shape_, other.shape_});
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
      auto tmp = *this;
      return tmp.sub_(other);
    }

    NDArray& sub_(const NDArray& other) {
      if (shape_ != other.shape_) {
        auto common_shape = get_common_shape(other);
        if (common_shape.empty()) {
          throw IncompatibleShapes("NDArray::sub", {shape_, other.shape_});
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

    NDArray& muls_(float s) {
      for (auto& v : arr_) {
        v *= s;
      }
      return *this;
    }

    NDArray muls(float s) const {
      auto tmp = *this;
      return tmp.muls_(s);
    }

    NDArray& divs_(float s) {
      return muls_(1.0f / s);
    }

    NDArray divs(float s) {
      auto tmp = *this;
      float recip = 1.0f / s;
      return tmp.muls_(recip);
    }

    NDArray mul(const NDArray& other) const {
      auto tmp = *this;
      return tmp.mul_(other);
    }

    NDArray& mul_(const NDArray& other) {
      if (shape_ != other.shape_) {
        auto common_shape = get_common_shape(other);
        if (common_shape.empty()) {
          throw IncompatibleShapes("NDArray::mul", {shape_, other.shape_});
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
      // same size
      if (shape_.size() != other.shape_.size()) {
        throw IncompatibleShapes("NDArray::dot", {shape_, other.shape_});
      }
      
      size_t axes = shape_.size();
      
      if (axes == 0) {
        return NDArray();
      }
      
      bool ok = true;
      for (size_t i = 0; ok && i < axes - 1; ++i) {
        if (shape_[i] != 1 || other.shape_[i] != 1) {
          ok = false;
        }
      }

      if (!ok || shape_[axes - 1] != other.shape_[axes - 1]) {
        throw IncompatibleShapes("NDArray::dot", {shape_, other.shape_});
      }

      NDArray res({1});
      for (size_t i = 0; i < arr_.size(); ++i) {
        res.arr_[0] += arr_[i] * other.arr_[i];
      }
      
      return res;
    }

    NDArray mm(const NDArray& other) const {
      // TODO: refactor shapes
      Shape shape1(shape_);
      Shape shape2(other.shape_);

      if (shape1.size() != 2 ||
          shape2.size() != 2 ||
          shape1[-1] != shape2[-2]) {
        throw IncompatibleShapes("NDArray::mm", {shape1.v(), shape2.v()});
      }

      // A=(m, n) B=(n, k) AB=(m, k)
      size_t m = shape1[-2];
      size_t n = shape1[-1];
      size_t k = shape2[-1];
      
      NDArray res({m, k});
      
      auto& A = arr_;
      auto& B = other.arr_;
      auto& AB = res.arr_;

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

      return res;
    }

    NDArray bmm(const NDArray& other) const {
      /*
       * Valid combinations:
       * - (m, n) * (b, n, k)  and  (b, m, n) * (n, k) = (b, m, k)
       * - (1, m, n) * (b, n, k)  and  (b, m, n) * (1, n, k) = (b, m, k)
       * - (b, m, n) * (b, n, k) = (b, m, k)
       */
      // TODO: refactor shapes
      Shape shape1(shape_);
      Shape shape2(other.shape_);
      
      size_t s1 = shape1.size();
      size_t s2 = shape2.size();

      bool ok = ((s1 == 2 && s2 == 3) || (s1 == 3 && s2 == 2) || (s1 == 3 && s2 == 3)) &&
                (shape1[-1] == shape2[-2]);

      if (ok && s1 == 3 && s2 == 3) {
        size_t s1_0 = shape1[-3];
        size_t s2_0 = shape2[-3];
        ok = s1_0 == s2_0 || s1_0 == 1 || s2_0 == 1;
      }

      if (!ok) {
        throw IncompatibleShapes("NDArray::bmm", {shape1.v(), shape2.v()});
      }

      // A=(?, m, n) B=(?, n, k) AB=(?, m, k)
      size_t m = shape1[-2];
      size_t n = shape1[-1];
      size_t k = shape2[-1];

      size_t c1 = s1 == 3 ? shape1[0] : 1;
      size_t c2 = s2 == 3 ? shape2[0] : 1;

      size_t count = std::max(c1, c2);
      
      NDArray res({count, m, k});
      
      for (size_t c = 0; c < count; ++c) {
        auto A = &arr_[c1 > 1 ? (c * m * n) : 0];
        auto B = &other.arr_[c2 > 1 ? (c * n * k) : 0];
        auto AB = &res.arr_[c * m * k];

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

  private:
    std::vector<float> arr_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
};


#endif // _ndarray_h_


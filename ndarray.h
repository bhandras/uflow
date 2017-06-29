#ifndef _ndarray_h_
#define _ndarray_h_

#include <vector>


class NDArray {
  public:
    NDArray(const std::vector<size_t>& shape)
      : shape_(shape) {
        update_shape();
    }

    void update_shape() {
      size_ = 1;
      for (auto dim : shape_) {
        size_ *= dim;
      }

      stride_ = std::vector<size_t>(1, shape_.size());
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
    }

  private:
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    size_t size_;
    std::vector<float> arr_;
};


#endif // _ndarray_h_


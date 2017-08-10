#ifndef _mnist_h_
#define _mnist_h_

#include <vector>
#include <fstream>
#include <random>
#include <cassert>

class MNIST {
  public:
    MNIST() 
      : train_idx_(train_size)
      , test_idx_(test_size) {
        for (int i = 0; i < train_size; ++i) {
          train_idx_[i] = i;
        }
        
        for (int i = 0; i < test_size; ++i) {
          test_idx_[i] = i;
        }
    }

    std::tuple<std::vector<float>, std::vector<int>> get_train_batch(size_t size) {
      std::random_device rd;
      std::default_random_engine re(rd());
      std::shuffle(train_idx_.begin(), train_idx_.end(), re);

      std::vector<float> data(size * img_size);
      std::vector<int> labels(size);

      for (size_t i = 0; i < size; ++i) {
        int idx = train_idx_[i];
        size_t offset = i * img_size;
        size_t img_offset = idx * img_size;
        
        std::copy(train_data_.begin() + img_offset, 
            train_data_.begin() + img_offset + img_size,
            data.begin() + offset);

        labels[i] = static_cast<int>(train_labels_[idx]);
      }

      return {data, labels};
    }

    void load(const std::string& path, bool normalize=true) {
      std::ifstream train_data_s(path + "/train-images-idx3-ubyte",
          std::ios::binary);
      
      if (train_data_s.fail()) {
        throw std::exception();
      }

      uint32_t magic, size, rows, cols;
      train_data_s.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
      train_data_s.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
      train_data_s.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
      train_data_s.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
   
      magic = msb_to_lsb(magic);
      size = msb_to_lsb(size);
      rows = msb_to_lsb(rows);
      cols = msb_to_lsb(cols);

      assert(magic == 0x00000803);
      assert(size == train_size);
      assert(rows * cols == img_size);

      size_t pos = 0;
      train_data_.resize(train_size * img_size);

      for (size_t i = 0; i < train_size; ++i) {
        unsigned char buf[img_size];
        train_data_s.read(reinterpret_cast<char*>(buf), img_size);
        
        for (size_t j = 0; j < img_size; ++j) {
          train_data_[pos] = static_cast<float>(buf[j]);
          if (normalize) {
            train_data_[pos] /= 255.0f;
          }
          ++pos;
        }
      }

      std::ifstream train_labels_s(path + "/train-labels-idx1-ubyte",
          std::ios::binary);
      train_labels_s.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
      train_labels_s.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
   
      magic = msb_to_lsb(magic);
      size = msb_to_lsb(size);

      assert(magic == 0x00000801);
      assert(size == train_size);
      train_labels_.reserve(train_size);
      train_labels_.assign((std::istreambuf_iterator<char>(train_labels_s)),
                           (std::istreambuf_iterator<char>()));
    }

    const size_t img_size = 28 * 28;
    const size_t train_size = 60000;
    const size_t test_size = 10000;
  
  private:
    uint32_t msb_to_lsb(uint32_t val) {
      return ((val >> 24) & 0xff)
        | ((val << 8) & 0xff0000)
        | ((val >> 8) & 0xff00)
        | ((val << 24) & 0xff000000);
    }

    std::random_device rd_;
 
    std::vector<int> train_idx_;
    std::vector<float> train_data_;
    std::vector<char> train_labels_;

    std::vector<int> test_idx_;
    std::vector<float> test_data_;
    std::vector<char> test_labels_;
};

#endif // _mnist_h_


#ifndef _util_h_
#define _util_h_

#include <vector>
#include <ostream>
#include <sstream>
#include <random>

template <class T>
std::string vstr(const std::vector<T>& v) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < v.size(); ++i) {
    ss << v[i];
    if (i != v.size() - 1) {
      ss << ",";
    }
  }
  ss << ")";
  return ss.str();
}


template <class T>
std::vector<T> random_vec(size_t size, T min_val, T max_val) {
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<T> distribution(min_val, max_val);
  std::vector<T> vec(size);
  
  std::generate(vec.begin(), vec.end(), [&](){
      return distribution(generator);
      });

  return vec;
}

std::vector<float> random_vec(size_t, float, float);


#endif // _util_h_


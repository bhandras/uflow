#ifndef _util_h_
#define _util_h_

#include <vector>
#include <sstream>

template <class T>
std::string str(const std::vector<T>& v) {
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

#endif // _util_h_


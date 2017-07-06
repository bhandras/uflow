#include "ndarray.h"

std::ostream& operator<<(std::ostream& os, const NDArray& arr) {
  os << arr.str();
  return os;
}

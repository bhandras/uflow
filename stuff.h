#ifndef _stuff_h_
#define _stuff_h_

#include <iostream>
#include <list>
#include <vector>
#include <queue>
#include <stack>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iterator>
#include <cctype>

#include <string>
#include <iomanip>
#include <iterator>
#include <stdexcept>

using namespace std;

// aliases
namespace test {

namespace impl {

static inline void _assert(const std::string& msg, bool val) {
  if (!val) {
    throw std::runtime_error(msg);
  }
}

} // namespace impl


class _assert_pred {
  public:
    enum predicate {
      eq,
      ne,
      lt,
      le,
      gt,
      ge
    };

    _assert_pred(predicate pred, const char* file, int line)
      : _pred(pred), _file(file), _line(line) {}
   
    template <class T1, class T2>
    void operator()(const T1& a, const T2& b) const {
      try {
        std::stringstream ss;
        ss <<  "expected: ";

        switch (_pred) {
          case eq:
            ss << a << " == " << b;
            impl::_assert(ss.str(), a == b);
            break;
          case ne:
            ss << a << " != " << b;
            impl::_assert(ss.str(), a != b);
            break;
          case lt:
            ss << a << " < " << b;
            impl::_assert(ss.str(), a < b);
            break;
          case le:
            ss << a << " <= " << b;
            impl::_assert(ss.str(), a <= b);
            break;
          case gt:
            ss << a << " > " << b;
            impl::_assert(ss.str(), a > b);
            break;
          case ge:
            ss << a << " >= " << b;
            impl::_assert(ss.str(), a >= b);
            break;
        }
      } catch (std::exception& e) {
        std::cout << _file << ":" << _line << " " << e.what() << std::endl;
      }
    }

  private:
    predicate _pred;
    const char* _file;
    const int _line;
};

class _assert_container_eq {
  public:
    _assert_container_eq(const char* file, int line)
      : _file(file), _line(line) {}
    
    template <class T>
    std::ostream& print(std::ostream& out, const T& c) const {
      out << '{' << c.size() << '}' << '['; 
      for (auto it = c.begin(); it != c.end(); ++it) {
        out << *it << ",";
      }
      
      if (!c.empty()) {
        out << "\b";
      }
      out << "]";
      
      return out;
    }

    template <class T1, class T2>
    void operator()(const T1& c1, const T2& c2) const {
      auto it1 = c1.begin();
      auto it2 = c2.begin();

      while (it1 != c1.end() && it2 != c2.end()) {
        if (*it1 != *it2) {
          break;
        }

        ++it1;
        ++it2;
      }

      if (it1 != c1.end() || it2 != c2.end()) {
        std::stringstream ss;
        ss << _file << ":" << _line << std::endl << std::setw(4);
        print(ss, c1);
        ss << std::endl << "!=" << std::endl << std::setw(4);
        print(ss, c2);
        std::cout << ss.str() << std::endl;
      }
    }

  private:
    const char* _file;
    const int _line;
};



} // namepsace test

#define assert_true(x) (::test::_assert_pred(test::_assert_pred::predicate::eq, __FILE__, __LINE__))((x), true)
#define assert_false(x) (::test::_assert_pred(test::_assert_pred::predicate::eq, __FILE__, __LINE__))((x), false)
#define assert_eq (::test::_assert_pred(test::_assert_pred::predicate::eq, __FILE__, __LINE__))
#define assert_ne (::test::_assert_pred(test::_assert_pred::predicate::ne, __FILE__, __LINE__))
#define assert_lt (::test::_assert_pred(test::_assert_pred::predicate::lt, __FILE__, __LINE__))
#define assert_le (::test::_assert_pred(test::_assert_pred::predicate::le, __FILE__, __LINE__))
#define assert_gt (::test::_assert_pred(test::_assert_pred::predicate::gt, __FILE__, __LINE__))
#define assert_ge (::test::_assert_pred(test::_assert_pred::predicate::ge, __FILE__, __LINE__))

#define assert_container_eq (::test::_assert_container_eq(__FILE__, __LINE__))

#


#endif // _stuff_h_

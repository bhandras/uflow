#ifndef _exception_h_
#define _exception_h_

#include <string>
#include <vector>
#include <exception>

#include "util.h"

class Exception : public std::exception {
  public:
    Exception(const std::string& type, const std::string& msg)
      : type_(type), msg_(msg) { }

    virtual const char* what() const throw() {
      return (type_ + " => " + msg_).c_str();
    }
  
  protected:
    std::string type_;
    std::string msg_;

};

class ValueError : public Exception {
  public:
    ValueError(const std::string& msg) 
      : Exception("ValueError", msg) { }
};

class RuntimeError : public Exception {
  public:
    RuntimeError(const std::string& msg)
      : Exception("RuntimeError", msg) { }
};

class IncompatibleShapes : public ValueError {
  public:
    static std::string helper(const std::string& fn,
        std::initializer_list<std::vector<size_t>> l) {
      std::stringstream ss;
      ss << fn << ": incompatible shapes {";

      for (auto& v : l) {
        ss << " " << vstr(v);
      }

      ss << " }";
      return ss.str();
    }
    
    IncompatibleShapes(const std::string& fn,
        std::initializer_list<std::vector<size_t>> shapes)
      : ValueError(helper(fn, shapes)) { }
};

#endif // _exception_h_


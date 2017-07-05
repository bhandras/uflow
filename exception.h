#ifndef _exception_h_
#define _exception_h_

#include <string>
#include <exception>

class ValueError : public std::exception {
  public:
    ValueError(const std::string& msg)
      : msg_(msg) { }

    virtual const char* what() const throw() {
      return ("ValueError => " + msg_).c_str();
    }
  
  private:
    std::string msg_;
};

#endif // _exception_h_


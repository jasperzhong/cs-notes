#ifndef MY_STRING_H
#define MY_STRING_H
#include <cstring>
#include <iostream>

namespace my_std {

class MyString {
 public:
  MyString() : ps_(nullptr), len_(0) {}

  MyString(const char* ps) {
    std::size_t len = 0;
    const char* p = ps;
    while (*p++ != '\0') len++;
    ps_ = new char[len + 1];
    strcpy(ps_, ps);
    len_ = len;
  }

  ~MyString() { delete[] ps_; }

  MyString(const MyString& s) {
    len_ = s.len_;
    ps_ = new char[len_ + 1];
    strcpy(ps_, s.ps_);
  }

  MyString& operator=(const MyString& s) {
    if (&s == this) return *this;
    delete[] ps_;
    ps_ = new char[s.len_ + 1];
    strcpy(ps_, s.ps_);
    return *this;
  }

  MyString(MyString&& s) noexcept : MyString() {
    using std::swap;
    swap(ps_, s.ps_);
    swap(len_, s.len_);
  }

  MyString& operator=(MyString&& s) noexcept {
    using std::swap;
    swap(ps_, s.ps_);
    swap(len_, s.len_);
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os, const MyString& s) {
    if (s.ps_) os << s.ps_;
    return os;
  }

  friend MyString operator+(const MyString& lhs, const MyString& rhs) {
    std::size_t len = lhs.len_ + rhs.len_;
    MyString s;
    s.len_ = len;
    s.ps_ = new char[len + 1];
    strcpy(s.ps_, lhs.ps_);
    strcpy(s.ps_ + lhs.len_, rhs.ps_);
    return s;
  }

 private:
  char* ps_;
  std::size_t len_;
};

}  // namespace my_std
#endif  // MY_STRING_H

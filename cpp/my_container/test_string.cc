#include <iostream>

#include "my_string.h"

using namespace my_std;

int main() {
  {
    MyString s1("Hello");
    MyString s2("World");
    MyString s3 = s1 + " " + s2 + "!";
    std::cout << "test1: s3=" << s3 << std::endl;
  }

  {
    // copy ctor
    MyString s1("Hello");
    MyString s2(s1);
    std::cout << "test2: s1=" << s1 << " s2=" << s2 << std::endl;
  }

  {
    // move ctor
    MyString s1("Hello");
    MyString s2 = std::move(s1);
    std::cout << "test3: s1=" << s1 << " s2=" << s2 << std::endl;
  }

  {
    MyString s1("Hello");
    MyString s2("World");
    s2 = std::move(s1);
    std::cout << "test4: s1=" << s1 << " s2=" << s2 << std::endl;
  }

  return 0;
}

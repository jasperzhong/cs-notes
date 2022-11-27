#ifndef MY_VECTOR_H
#define MY_VECTOR_H

#include <cstdlib>
#include <iostream>
#include <utility>

namespace my_std {

template <typename T>
class MyVector {
 public:
  MyVector() : data_(nullptr), size_(0), capacity_(0){};

  explicit MyVector(std::size_t n) : size_(n), capacity_(n) {
    data_ = new T[n]();
  };

  MyVector(std::size_t n, const T& value) : MyVector(n) {
    for (std::size_t i = 0; i < n; i++) data_[i] = value;
  }

  ~MyVector() { delete[] data_; }

  MyVector(const MyVector& v) : size_(v.size_), capacity_(v.capacity_) {
    data_ = new T[capacity_];
    for (std::size_t i = 0; i < size_; i++) data_[i] = v.data_[i];
  }

  MyVector& operator=(const MyVector& v) {
    if (&v == this) return *this;
    delete[] data_;
    size_ = v.size_;
    capacity_ = v.capacity_;
    v.data_ = new T[capacity_];
    for (std::size_t i = 0; i < size_; i++) data_[i] = v.data_[i];
    return *this;
  }

  MyVector(MyVector&& v) noexcept : MyVector() {
    using std::swap;
    swap(data_, v.data_);
    swap(size_, v.size_);
    swap(capacity_, v.capacity_);
  }

  MyVector& operator=(MyVector&& v) noexcept {
    using std::swap;
    swap(data_, v.data_);
    swap(size_, v.size_);
    swap(capacity_, v.capacity_);
    return *this;
  }

  T& operator[](std::size_t index) { return data_[index]; }

  const T& operator[](std::size_t index) const {
    return const_cast<T&>(static_cast<const MyVector&>(*this)[index]);
  }

  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }
  bool empty() const { return size_ == 0; }

  void clear() { size_ = 0; }

  void push_back(const T& value) {
    if (size_ >= capacity_) {
      reallocate();
    }
    data_[size_++] = value;
  }

  void push_back(T&& value) {
    if (size_ >= capacity_) {
      reallocate();
    }
    data_[size_++] = std::move(value);
  }

  template <class... Args>
  void emplace_back(Args&&... args) {
    if (size_ >= capacity_) {
      reallocate();
    }
    data_[size_++] = T(std::forward<Args>(args)...);
  }

 private:
  void reallocate() {
    capacity_ = capacity_ == 0 ? 1 : capacity_ * 2;
    auto data = new T[capacity_];
    for (std::size_t i = 0; i < size_; ++i) {
      data[i] = std::move(data_[i]);
    }
    delete[] data_;
    data_ = data;
  }

  T* data_;
  std::size_t size_;
  std::size_t capacity_;
};

}  // namespace my_std

#endif  // MY_VECTOR_H

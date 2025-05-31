#include "Utilities/FileReader.h"
#include <cassert>

namespace neural_network {

FileReader::FileReader(const std::filesystem::path &file) : file_(file) {
  assert(file_.is_open() && "Failed to open file for reading");
}

template <class T> FileReader &FileReader::operator>>(T &x) {
  file_ >> x;
  return *this;
}

// Instantiate explicitly for common usage
template FileReader &FileReader::operator>>(int &);
template FileReader &FileReader::operator>>(double &);

template <class T, class A>
FileReader &operator>>(FileReader &r, std::vector<T, A> &vec) {
  size_t size;
  r >> size;
  vec.resize(size);
  for (size_t i = 0; i < size; ++i)
    r >> vec[i];
  return r;
}

} // namespace neural_network

#include "Utilities/FileWriter.h"
#include <cassert>

namespace neural_network {

FileWriter::FileWriter(const std::filesystem::path &file) : file_(file) {
  assert(file_.is_open() && "Failed to open file for writing");
}

template <class T> FileWriter &FileWriter::operator<<(const T &x) {
  file_ << x << ' ';
  return *this;
}

// Instantiate explicitly for common usage
template FileWriter &FileWriter::operator<<(const int &);
template FileWriter &FileWriter::operator<<(const double &);

template <class T, class A>
FileWriter &operator<<(FileWriter &w, const std::vector<T, A> &vec) {
  w << vec.size();
  for (const auto &x : vec)
    w << x;
  return w;
}

} // namespace neural_network

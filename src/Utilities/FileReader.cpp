#include "Utilities/FileReader.h"
#include "Layers/Layer.h"
#include <stdexcept>

namespace neural_network {

FileReader::FileReader(const std::filesystem::path &path) : in_(path) {
  if (!in_) {
    throw std::runtime_error("Failed to open file for reading");
  }
}

template <typename T> FileReader &operator>>(FileReader &r, T &value) {
  value.read(r);
  return r;
}

FileReader &operator>>(FileReader &r, Vector &v) {
  Index size;
  r >> size;
  v.resize(size);
  for (Index i = 0; i < size; ++i)
    r >> v(i);
  return r;
}

FileReader &operator>>(FileReader &r, Matrix &m) {
  Index rows, cols;
  r >> rows >> cols;
  m.resize(rows, cols);
  for (Index i = 0; i < rows; ++i)
    for (Index j = 0; j < cols; ++j)
      r >> m(i, j);
  return r;
}

template <> FileReader &operator>>(FileReader &r, std::vector<Layer> &v) {
  Index size;
  r >> size;
  v.resize(size);
  for (auto &e : v)
    e.read(r);
  return r;
}

// Explicit instantiations
template FileReader &operator>>(FileReader &, int &);
template FileReader &operator>>(FileReader &, size_t &);
template FileReader &operator>>(FileReader &, double &);
template FileReader &operator>>(FileReader &, std::vector<Layer> &);

} // namespace neural_network

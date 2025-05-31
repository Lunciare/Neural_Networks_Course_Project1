#include "Utilities/FileReader.h"
#include <stdexcept>

namespace neural_network {

FileReader::FileReader(const std::string &filename) : in_(filename) {
  if (!in_) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
}

// Перегрузки для базовых типов
FileReader &operator>>(FileReader &r, int &value) {
  r.in_ >> value;
  return r;
}

FileReader &operator>>(FileReader &r, size_t &value) {
  r.in_ >> value;
  return r;
}

FileReader &operator>>(FileReader &r, long &value) {
  r.in_ >> value;
  return r;
}

FileReader &operator>>(FileReader &r, double &value) {
  r.in_ >> value;
  return r;
}

// Перегрузка для Eigen-вектора
FileReader &operator>>(FileReader &r, Vector &v) {
  size_t size;
  r >> size;
  v.resize(size);
  for (Index i = 0; i < v.size(); ++i)
    r.in_ >> v(i);
  return r;
}

// Перегрузка для Eigen-матрицы
FileReader &operator>>(FileReader &r, Matrix &m) {
  Index rows, cols;
  r >> rows >> cols;
  m.resize(rows, cols);
  for (Index i = 0; i < rows; ++i)
    for (Index j = 0; j < cols; ++j)
      r.in_ >> m(i, j);
  return r;
}

// Перегрузка для вектора слоёв
FileReader &operator>>(FileReader &r, std::vector<Layer> &v) {
  size_t size;
  r >> size;
  v.resize(size);
  for (auto &layer : v)
    layer.read(r);
  return r;
}

} // namespace neural_network

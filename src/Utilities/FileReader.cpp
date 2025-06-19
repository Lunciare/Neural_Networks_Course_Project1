#include "Utilities/FileReader.h"
#include "Layers/Layer.h"
#include "Model/Model.h"

#include <stdexcept>

namespace neural_network {

FileReader::FileReader(const std::filesystem::path &file) {
  file_.open(file, std::ios::in);
  if (!file_.is_open()) {
    throw std::runtime_error("Could not open file for reading.");
  }
}

FileReader::~FileReader() { file_.close(); }

FileReader &operator>>(FileReader &r, Vector &v) {
  Index size;
  r >> size;
  v.resize(size);
  for (Index i = 0; i < size; ++i)
    r >> v[i];
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

FileReader &operator>>(FileReader &r, Layer &l) {
  r >> l.weights_ >> l.biases_;
  int type;
  r >> type;
  l.activation_type_ = static_cast<ActivationFunction::Type>(type);
  l.activation_ = ActivationFunction::create(l.activation_type_);
  return r;
}

FileReader &operator>>(FileReader &r, Model &m) {
  size_t n;
  r >> n;
  m.layers_.resize(n);
  for (auto &layer : m.layers_) {
    r >> layer;
  }
  return r;
}

} // namespace neural_network

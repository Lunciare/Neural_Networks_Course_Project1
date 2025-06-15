#include "Utilities/FileWriter.h"
#include "Layers/Layer.h"
#include "Model/Model.h"

#include <stdexcept>

namespace neural_network {

FileWriter::FileWriter(const std::filesystem::path &file) {
  file_.open(file, std::ios::out);
  if (!file_.is_open()) {
    throw std::runtime_error("Could not open file for writing.");
  }
}

FileWriter::~FileWriter() { file_.close(); }

FileWriter &operator<<(FileWriter &w, const Vector &v) {
  w << v.size();
  for (Index i = 0; i < v.size(); ++i)
    w << v[i];
  return w;
}

FileWriter &operator<<(FileWriter &w, const Matrix &m) {
  w << m.rows() << m.cols();
  for (Index i = 0; i < m.rows(); ++i)
    for (Index j = 0; j < m.cols(); ++j)
      w << m(i, j);
  return w;
}

FileWriter &operator<<(FileWriter &w, const Layer &l) {
  w << l.weights_ << l.biases_;
  w << static_cast<int>(l.activation_type_);
  return w;
}

FileWriter &operator<<(FileWriter &w, const Model &m) {
  w << m.layers_.size();
  for (const auto &layer : m.layers_) {
    w << layer;
  }
  return w;
}

} // namespace neural_network

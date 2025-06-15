#include "Utilities/FileReader.h"
#include "Layers/Layer.h"
#include "Model/Model.h"
#include <cassert>

namespace neural_network {

FileReader::FileReader(const std::filesystem::path &file) {
  file_.open(file);
  if (!file_.is_open()) {
    throw std::runtime_error("Could not open file for reading.");
  }
}

FileReader::~FileReader() { file_.close(); }

template <> FileReader &FileReader::operator>>(std::string &s) {
  std::getline(file_, s);
  return *this;
}

template <typename T> FileReader &operator>>(FileReader &r, std::vector<T> &v) {
  size_t size;
  r >> size;
  v.resize(size);
  for (size_t i = 0; i < size; ++i) {
    r >> v[i];
  }
  return r;
}

FileReader &operator>>(FileReader &r, Model &m) { return r >> m.layers_; }

FileReader &operator>>(FileReader &r, Layer &l) {
  return r >> l.weights_ >> l.biases_ >> l.activation_;
}

} // namespace neural_network

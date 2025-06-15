#include "Utilities/FileWriter.h"
#include "Layers/Layer.h"
#include "Model/Model.h"
#include <cassert>

namespace neural_network {

FileWriter::FileWriter(const std::filesystem::path &file) {
  file_.open(file);
  if (!file_.is_open()) {
    throw std::runtime_error("Could not open file for writing.");
  }
}

FileWriter::~FileWriter() { file_.close(); }

template <> FileWriter &FileWriter::operator<<(const std::string &s) {
  file_ << s << '\n';
  return *this;
}

template <typename T>
FileWriter &operator<<(FileWriter &w, const std::vector<T> &v) {
  w << v.size();
  for (const auto &item : v) {
    w << item;
  }
  return w;
}

FileWriter &operator<<(FileWriter &w, const Model &m) { return w << m.layers_; }

FileWriter &operator<<(FileWriter &w, const Layer &l) {
  return w << l.weights_ << l.biases_ << l.activation_;
}

} // namespace neural_network

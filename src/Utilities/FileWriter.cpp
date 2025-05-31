// FileWriter.cpp
#include "Utilities/FileWriter.h"
#include "Layers/Layer.h"

namespace neural_network {

FileWriter::FileWriter(const std::filesystem::path &path) : out_(path) {
  if (!out_)
    throw std::runtime_error("Failed to open file for writing");
}

template <typename T> FileWriter &FileWriter::operator<<(const T &value) {
  out_ << value << '\n';
  return *this;
}

FileWriter &operator<<(FileWriter &w, const Vector &v) {
  w.out_ << v.size() << '\n';
  for (Index i = 0; i < v.size(); ++i)
    w.out_ << v(i) << ' ';
  w.out_ << '\n';
  return w;
}

FileWriter &operator<<(FileWriter &w, const Matrix &m) {
  w.out_ << m.rows() << ' ' << m.cols() << '\n';
  for (Index i = 0; i < m.rows(); ++i) {
    for (Index j = 0; j < m.cols(); ++j)
      w.out_ << m(i, j) << ' ';
    w.out_ << '\n';
  }
  return w;
}

FileWriter &operator<<(FileWriter &w, const Layer &layer) {
  layer.write(w); // вызывает Layer::write<FileWriter>
  return w;
}

template <typename T>
FileWriter &operator<<(FileWriter &w, const std::vector<T> &v) {
  w << static_cast<Index>(v.size());
  for (const auto &e : v)
    w << e;
  return w;
}

FileWriter &operator<<(FileWriter &w, const std::vector<Layer> &v) {
  w << static_cast<Index>(v.size());
  for (const auto &layer : v) {
    layer.write(w);
  }
  return w;
}

// explicit instantiations
template FileWriter &FileWriter::operator<<(const int &);
template FileWriter &FileWriter::operator<<(const double &);
template FileWriter &FileWriter::operator<<(const std::string &);

} // namespace neural_network

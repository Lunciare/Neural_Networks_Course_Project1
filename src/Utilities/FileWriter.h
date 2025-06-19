#pragma once

#include "Utilities/Utils.h"
#include <filesystem>
#include <fstream>
#include <vector>

namespace neural_network {

class Model;
class Layer;

class FileWriter {
public:
  explicit FileWriter(const std::filesystem::path &file);
  ~FileWriter();

  template <typename T> FileWriter &operator<<(const T &x) {
    file_ << x << '\n';
    return *this;
  }

private:
  std::ofstream file_;
};

FileWriter &operator<<(FileWriter &w, const Vector &v);
FileWriter &operator<<(FileWriter &w, const Matrix &m);
FileWriter &operator<<(FileWriter &w, const Layer &l);
FileWriter &operator<<(FileWriter &w, const Model &m);

} // namespace neural_network

#pragma once

#include "Utilities/Utils.h"
#include <filesystem>
#include <fstream>
#include <vector>

namespace neural_network {

class Model;
class Layer;

class FileReader {
public:
  explicit FileReader(const std::filesystem::path &file);
  ~FileReader();

  template <typename T> FileReader &operator>>(T &x) {
    file_ >> x;
    return *this;
  }

private:
  std::ifstream file_;
};

FileReader &operator>>(FileReader &r, Vector &v);
FileReader &operator>>(FileReader &r, Matrix &m);
FileReader &operator>>(FileReader &r, Layer &l);
FileReader &operator>>(FileReader &r, Model &m);

} // namespace neural_network

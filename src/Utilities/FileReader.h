#pragma once

#include "Utilities/Utils.h"
#include <filesystem>
#include <fstream>
#include <vector>

namespace neural_network {

class FileReader {
public:
  explicit FileReader(const std::filesystem::path &path);

  template <typename T> friend FileReader &operator>>(FileReader &r, T &value);

  friend FileReader &operator>>(FileReader &r, Vector &v);
  friend FileReader &operator>>(FileReader &r, Matrix &m);

  template <typename T>
  friend FileReader &operator>>(FileReader &r, std::vector<T> &v);

private:
  std::ifstream in_;
};

} // namespace neural_network

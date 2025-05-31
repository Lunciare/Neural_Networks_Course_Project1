#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "Layers/Layer.h"
#include "Utilities/Utils.h"

namespace neural_network {

class FileReader {
public:
  explicit FileReader(const std::string &filename);
  std::ifstream in_;
};

FileReader &operator>>(FileReader &r, int &value);
FileReader &operator>>(FileReader &r, size_t &value);
FileReader &operator>>(FileReader &r, long &value);
FileReader &operator>>(FileReader &r, double &value);

FileReader &operator>>(FileReader &r, Vector &v);
FileReader &operator>>(FileReader &r, Matrix &m);

FileReader &operator>>(FileReader &r, std::vector<Layer> &v);

} // namespace neural_network

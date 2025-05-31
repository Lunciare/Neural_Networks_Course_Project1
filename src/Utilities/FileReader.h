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

// Перегрузки для стандартных типов
FileReader &operator>>(FileReader &r, int &value);
FileReader &operator>>(FileReader &r, size_t &value);
FileReader &operator>>(FileReader &r, long &value);
FileReader &operator>>(FileReader &r, double &value);

// Перегрузки для Eigen-векторов и матриц
FileReader &operator>>(FileReader &r, Vector &v);
FileReader &operator>>(FileReader &r, Matrix &m);

// Перегрузка для вектора слоёв
FileReader &operator>>(FileReader &r, std::vector<Layer> &v);

} // namespace neural_network

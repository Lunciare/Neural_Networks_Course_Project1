#pragma once
#include "Utilities/Utils.h"
#include <filesystem>
#include <fstream>

namespace neural_network {

class FileWriter {
public:
  explicit FileWriter(const std::filesystem::path &path);
  template <typename T> FileWriter &operator<<(const T &value);

  std::ofstream out_;
};

// Специализации
FileWriter &operator<<(FileWriter &w, const Vector &v);
FileWriter &operator<<(FileWriter &w, const Matrix &m);
FileWriter &operator<<(FileWriter &w,
                       const class Layer &layer); // 👈 добавь это
template <typename T>
FileWriter &operator<<(FileWriter &w, const std::vector<T> &v);

} // namespace neural_network

#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace neural_network {

class FileWriter {
public:
  explicit FileWriter(const std::filesystem::path &file);
  ~FileWriter();

  template <typename T> FileWriter &operator<<(const T &x);

private:
  std::ofstream file_;
};

template <typename T> FileWriter &FileWriter::operator<<(const T &x) {
  file_ << x << '\n';
  return *this;
}

template <typename T>
FileWriter &operator<<(FileWriter &w, const std::vector<T> &v);

FileWriter &operator<<(FileWriter &w, const Model &m);
FileWriter &operator<<(FileWriter &w, const Layer &l);

} // namespace neural_network

#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace neural_network {

class FileReader {
public:
  explicit FileReader(const std::filesystem::path &file);
  ~FileReader();

  template <typename T> FileReader &operator>>(T &x);

private:
  std::ifstream file_;
};

template <typename T> FileReader &FileReader::operator>>(T &x) {
  file_ >> x;
  return *this;
}

template <typename T> FileReader &operator>>(FileReader &r, std::vector<T> &v);

FileReader &operator>>(FileReader &r, Model &m);
FileReader &operator>>(FileReader &r, Layer &l);

} // namespace neural_network

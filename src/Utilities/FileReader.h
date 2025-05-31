#pragma once
#include <fstream>
#include <vector>

namespace neural_network {

class FileReader {
public:
  explicit FileReader(const std::string &filename);
  ~FileReader();

  template <typename T> FileReader &operator>>(T &data);

  bool good() const;

private:
  std::ifstream file_;
};

template <typename T> FileReader &FileReader::operator>>(T &data) {
  file_ >> data;
  return *this;
}

} // namespace neural_network

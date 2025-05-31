#pragma once
#include <fstream>
#include <vector>

namespace neural_network {

class FileWriter {
public:
  explicit FileWriter(const std::string &filename);
  ~FileWriter();

  template <typename T> FileWriter &operator<<(const T &data);

  bool good() const;

private:
  std::ofstream file_;
};

template <typename T> FileWriter &FileWriter::operator<<(const T &data) {
  file_ << data << ' ';
  return *this;
}

} // namespace neural_network

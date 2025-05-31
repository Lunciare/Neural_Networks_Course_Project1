#pragma once
#include <filesystem>
#include <fstream>
#include <vector>

namespace neural_network {

class FileWriter {
public:
  explicit FileWriter(const std::filesystem::path &file);

  template <class T> FileWriter &operator<<(const T &x);

private:
  std::ofstream file_;
};

template <class T, class A>
FileWriter &operator<<(FileWriter &w, const std::vector<T, A> &vec);

} // namespace neural_network

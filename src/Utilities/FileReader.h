#pragma once
#include <fstream>
#include <vector>

namespace neural_network {

class FileReader {
public:
  explicit FileReader(const std::filesystem::path &file);

  template <class T> FileReader &operator>>(T &x);

private:
  std::ifstream file_;
};

template <class T, class A>
FileReader &operator>>(FileReader &r, std::vector<T, A> &vec);

} // namespace neural_network

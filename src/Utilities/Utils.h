#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace neural_network {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;

template <typename T, typename Tag> class StrongAlias {
public:
  explicit StrongAlias(T value) : value_(value) {}

  operator T() const { return value_; }

  T get() const { return value_; }

private:
  T value_;
};

struct InTag {};
struct OutTag {};

using In = StrongAlias<Index, InTag>;
using Out = StrongAlias<Index, OutTag>;

template <typename T> Index size(const std::vector<T> &vec) {
  return static_cast<Index>(vec.size());
}

} // namespace neural_network

#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace neural_network {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;
struct In {
  Index value;
  explicit In(Index v) : value(v) {}
  operator Index() const { return value; }
};
struct Out {
  Index value;
  explicit Out(Index v) : value(v) {}
  operator Index() const { return value; }
};
} // namespace neural_network

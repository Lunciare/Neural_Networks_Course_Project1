#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace neural_network {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;

class In {
public:
  explicit In(Index v) : value_(v) {}
  operator Index() const { return value_; }

private:
  Index value_;
};

class Out {
public:
  explicit Out(Index v) : value_(v) {}
  operator Index() const { return value_; }

private:
  Index value_;
};

} // namespace neural_network

#pragma once

#include <Eigen/Dense>
#include <functional>

#include "Math.h"

namespace neural_network {
class LossFunction {
  using Signature1 = std::function<double(const Matrix &, const Matrix &)>;
  using Signature2 = std::function<Matrix(const Matrix &, const Matrix &)>;
  LossFunction(Signature1 f0, Signature2 f1);

public:
  static LossFunction Euclid();
  static LossFunction CrossEntropy();
  static LossFunction CrossEntropyWithLogits();

  double dist(const Matrix &x, const Matrix &y) const;
  Matrix derDist(const Matrix &x, const Matrix &y) const;

private:
  Signature1 f0_;
  Signature2 f1_;
};

} // namespace neural_network

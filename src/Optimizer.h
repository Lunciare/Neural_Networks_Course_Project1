#pragma once

#include <Eigen/Dense>
#include <functional>

#include "Math.h"

namespace neural_network {
class Optimizer {
  // using Signature = std::function<void(const Matrix&, Matrix*)>;

  using MSignature =
      std::function<Matrix(const Matrix &grad, const Matrix &weights,
                           Matrix &memory, Index time_step)>;
  using VSignature =
      std::function<Vector(const Vector &grad, const Vector &weights,
                           Vector &memory, Index time_step)>;

  Optimizer(MSignature optimizerA, VSignature optimizerB);

public:
  static Optimizer SGD(double learning_rate);
  static Optimizer Momentum(double learning_rate, double beta1);
  static Optimizer Adam(double learning_rate, double beta1, double beta2,
                        double epsilon = 1e-8);

  Matrix getUpdateA(const Matrix &grad, const Matrix &cur_w,
                    Matrix &memory = dMatrix(), Index time_step = 1) const;

  Vector getUpdateB(const Vector &grad, const Vector &cur_w,
                    Vector &memory = dVector(), Index time_step = 1) const;

private:
  MSignature optimizerA_;
  VSignature optimizerB_;

  static Matrix &dMatrix();
  static Vector &dVector();
};

} // namespace neural_network

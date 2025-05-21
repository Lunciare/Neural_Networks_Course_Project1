#pragma once

#include "Utilities/LinAlg.h"

namespace neural_network {

// Supported optimizer types
enum class OptimizerType { SGD, Adam };

// Optimizer class: applies SGD or Adam update rule
class Optimizer {
public:
  // Constructor: default is Adam
  Optimizer(double lr, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-8, OptimizerType type = OptimizerType::Adam);

  // Update parameter matrix in place
  void update(Matrix &param, Matrix &m, Matrix &v, const Matrix &grad,
              Index t) const;

  // Update parameter vector in place
  void update(Vector &param, Vector &m, Vector &v, const Vector &grad,
              Index t) const;

  // Get optimizer type
  OptimizerType type() const { return type_; }

private:
  double lr_;    // Learning rate
  double beta1_; // Exponential decay for first moment
  double beta2_; // Exponential decay for second moment
  double eps_;   // Small value for numerical stability
  OptimizerType type_;
};

} // namespace neural_network

#pragma once
#include "Utilities/LinAlg.h"

namespace neural_network {

// Provides static methods for loss computation (e.g. mean squared error).
class LossFunction {
public:
  // Computes the Mean Squared Error (MSE) between prediction and target column
  // vectors.
  static double mse(const Vector &y_pred, const Vector &y_true);

  // Computes the gradient of MSE with respect to y_pred (both column vectors).
  static Vector mseGrad(const Vector &y_pred, const Vector &y_true);
};

} // namespace neural_network

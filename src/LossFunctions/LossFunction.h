#pragma once
#include "Utilities/Utils.h"

namespace neural_network {

class LossFunction {
public:
  static double mse(const Vector &y_pred, const Vector &y_true);
  static Vector mseGrad(const Vector &y_pred, const Vector &y_true);

  static double crossEntropy(const Vector &y_pred, const Vector &y_true);
  static Vector crossEntropyGrad(const Vector &y_pred, const Vector &y_true);
};

} // namespace neural_network

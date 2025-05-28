#pragma once
#include "Utilities/LinAlg.h"

namespace neural_network {

class ActivationFunction {
public:
  enum class Type { ReLU, Sigmoid, Identity, Tanh };

  // Applies the activation function (elementwise) to the input vector.
  static Vector apply(Type type, const Vector &x);

  // Computes the derivative of the activation function (elementwise), given the
  // activation output.
  static Vector derivative(Type type, const Vector &y);
};

} // namespace neural_network

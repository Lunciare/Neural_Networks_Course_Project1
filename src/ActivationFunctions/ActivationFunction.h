#pragma once
#include "Utilities/Utils.h"
#include <cassert>
#include <functional>

namespace neural_network {

class ActivationFunction {
public:
  enum class Type { ReLU, Sigmoid, Identity, Tanh };

  using Function = std::function<Vector(const Vector &)>;

  ActivationFunction(Function f_apply, Function f_derivative);

  Vector apply(const Vector &x) const;
  Vector derivative(const Vector &x) const;

  static ActivationFunction ReLU();
  static ActivationFunction Sigmoid();
  static ActivationFunction Identity();
  static ActivationFunction Tanh();

  static ActivationFunction create(Type type);

private:
  Function f_apply_;
  Function f_derivative_;
};

} // namespace neural_network

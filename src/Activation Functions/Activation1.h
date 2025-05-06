#pragma once

#include <functional>

#include "Math.h"

namespace neural_network {

class ActivationFunction {
  using Signature = double(double);
  using Function = std::function<Signature>;
  ActivationFunction(Function f0, Function f1);

public:
  static ActivationFunction ReLU();
  static ActivationFunction Sigmoid();
  static ActivationFunction Identity();
  static ActivationFunction Tanh();

  double evaluate(double x) const;
  double derEvaluate(double x) const;

  Matrix evaluate(const Matrix &x) const;
  Matrix derEvaluate(const Matrix &x) const;

private:
  Function f0_; // функция
  Function f1_; // производная
};

} // namespace neural_network

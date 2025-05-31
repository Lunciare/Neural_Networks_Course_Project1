#include "ActivationFunctions/ActivationFunction.h"
#include <cassert>
#include <cmath>

namespace neural_network {

using Type = ActivationFunction::Type;

ActivationFunction::ActivationFunction(Function f_apply, Function f_derivative)
    : f_apply_(std::move(f_apply)), f_derivative_(std::move(f_derivative)) {}

Vector ActivationFunction::apply(const Vector &x) const { return f_apply_(x); }

Vector ActivationFunction::derivative(const Vector &x) const {
  return f_derivative_(x);
}

ActivationFunction ActivationFunction::ReLU() {
  return ActivationFunction(
      [](const Vector &x) { return x.array().max(0.0).matrix(); },
      [](const Vector &x) {
        return (x.array() > 0.0).cast<double>().matrix();
      });
}

ActivationFunction ActivationFunction::Sigmoid() {
  return ActivationFunction(
      [](const Vector &x) {
        return (1.0 / (1.0 + (-x.array()).exp())).matrix();
      },
      [](const Vector &x) {
        Vector sig = (1.0 / (1.0 + (-x.array()).exp())).matrix();
        return (sig.array() * (1.0 - sig.array())).matrix();
      });
}

ActivationFunction ActivationFunction::Identity() {
  return ActivationFunction(
      [](const Vector &x) { return x; },
      [](const Vector &x) { return Vector::Ones(x.size()); });
}

ActivationFunction ActivationFunction::Tanh() {
  return ActivationFunction(
      [](const Vector &x) { return x.array().tanh().matrix(); },
      [](const Vector &x) {
        return (1.0 - x.array().tanh().square()).matrix();
      });
}

ActivationFunction ActivationFunction::create(Type type) {
  switch (type) {
  case Type::ReLU:
    return ReLU();
  case Type::Sigmoid:
    return Sigmoid();
  case Type::Identity:
    return Identity();
  case Type::Tanh:
    return Tanh();
  default:
    assert(false && "Unknown ActivationFunction::Type");
    return ReLU(); // fallback
  }
}

} // namespace neural_network

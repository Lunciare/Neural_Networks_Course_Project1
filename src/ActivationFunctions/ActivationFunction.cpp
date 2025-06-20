#include "ActivationFunctions/ActivationFunction.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

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
        return x.unaryExpr([](double v) {
          const double s = 1.0 / (1.0 + std::exp(-v));
          return s * (1.0 - s);
        });
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

ActivationFunction ActivationFunction::Softmax() {
  auto apply = [](const Vector &x) -> Vector {
    Vector shifted = x.array() - x.maxCoeff();
    Vector exps = shifted.array().exp();
    return exps / exps.sum();
  };

  auto derivative = [](const Vector &x) -> Vector {
    return Vector::Ones(x.size());
  };

  return ActivationFunction(apply, derivative);
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
  case Type::Softmax:
    return Softmax();
  default:
    assert(false && "Unknown ActivationFunction::Type");
    return ReLU();
  }
}

} // namespace neural_network

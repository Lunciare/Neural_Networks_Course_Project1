#include "ActivationFunctions/ActivationFunction.h"
#include <cassert>
#include <cmath>

namespace neural_network {

Vector ActivationFunction::apply(Type type, const Vector &x) {
  switch (type) {
  case Type::ReLU:
    return x.array().max(0.0).matrix();
  case Type::Sigmoid:
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
  case Type::Identity:
    return x;
  case Type::Tanh:
    return x.array().tanh().matrix();
  default:
    assert(false && "Unknown activation type");
    return x;
  }
}

Vector ActivationFunction::derivative(Type type, const Vector &y) {
  switch (type) {
  case Type::ReLU:
    return (y.array() > 0.0).cast<double>().matrix();
  case Type::Sigmoid:
    return (y.array() * (1.0 - y.array())).matrix();
  case Type::Identity:
    return Vector::Ones(y.size());
  case Type::Tanh:
    return (1.0 - y.array().square()).matrix();
  default:
    assert(false && "Unknown activation type");
    return y;
  }
}

} // namespace neural_network

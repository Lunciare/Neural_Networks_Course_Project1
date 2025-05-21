#include "LossFunctions/LossFunction.h"
#include <cassert>

namespace neural_network {

double LossFunction::mse(const Vector &y_pred, const Vector &y_true) {
  assert(y_pred.size() == y_true.size());
  auto diff = y_pred - y_true;
  return diff.squaredNorm() / diff.size();
}

Vector LossFunction::mseGrad(const Vector &y_pred, const Vector &y_true) {
  assert(y_pred.size() == y_true.size());
  return (2.0 / y_pred.size()) * (y_pred - y_true);
}

} // namespace neural_network

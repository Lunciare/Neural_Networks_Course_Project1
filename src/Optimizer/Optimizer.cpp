#include "Optimizer/Optimizer.h"
#include <cmath>

namespace neural_network {

Optimizer::Optimizer(double lr, double beta1, double beta2, double eps,
                     OptimizerType type)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), type_(type) {}

void Optimizer::update(Matrix &param, Matrix &m, Matrix &v, const Matrix &grad,
                       Index t) const {
  if (type_ == OptimizerType::SGD) {
    param -= lr_ * grad;
  } else if (type_ == OptimizerType::Adam) {
    m = beta1_ * m + (1.0 - beta1_) * grad;
    v = beta2_ * v + (1.0 - beta2_) * grad.array().square().matrix();
    Matrix m_hat = m / (1.0 - std::pow(beta1_, t));
    Matrix v_hat = v / (1.0 - std::pow(beta2_, t));
    param -= lr_ * (m_hat.array() / (v_hat.array().sqrt() + eps_)).matrix();
  }
}

void Optimizer::update(Vector &param, Vector &m, Vector &v, const Vector &grad,
                       Index t) const {
  if (type_ == OptimizerType::SGD) {
    param -= lr_ * grad;
  } else if (type_ == OptimizerType::Adam) {
    m = beta1_ * m + (1.0 - beta1_) * grad;
    v = beta2_ * v + (1.0 - beta2_) * grad.array().square().matrix();
    Vector m_hat = m / (1.0 - std::pow(beta1_, t));
    Vector v_hat = v / (1.0 - std::pow(beta2_, t));
    param -= lr_ * (m_hat.array() / (v_hat.array().sqrt() + eps_)).matrix();
  }
}

} // namespace neural_network

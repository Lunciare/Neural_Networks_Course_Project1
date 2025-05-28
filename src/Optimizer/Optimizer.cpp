#include "Optimizer/Optimizer.h"
#include <cmath>

namespace neural_network {

Optimizer::Optimizer(
    std::function<void(Matrix &, Matrix &, Matrix &, const Matrix &, Index)> um,
    std::function<void(Vector &, Vector &, Vector &, const Vector &, Index)> uv)
    : update_matrix_(std::move(um)), update_vector_(std::move(uv)) {}

Optimizer Optimizer::SGD(double lr) {
  auto update_m = [lr](Matrix &p, Matrix &, Matrix &, const Matrix &g, Index) {
    p -= lr * g;
  };
  auto update_v = [lr](Vector &p, Vector &, Vector &, const Vector &g, Index) {
    p -= lr * g;
  };
  return Optimizer(update_m, update_v);
}

Optimizer Optimizer::Adam(double lr, double beta1, double beta2, double eps) {
  auto update_m = [=](Matrix &p, Matrix &m, Matrix &v, const Matrix &g,
                      Index t) {
    m = beta1 * m + (1.0 - beta1) * g;
    v = beta2 * v + (1.0 - beta2) * g.array().square().matrix();
    Matrix m_hat = m / (1.0 - std::pow(beta1, t));
    Matrix v_hat = v / (1.0 - std::pow(beta2, t));
    p -= lr * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();
  };
  auto update_v = [=](Vector &p, Vector &m, Vector &v, const Vector &g,
                      Index t) {
    m = beta1 * m + (1.0 - beta1) * g;
    v = beta2 * v + (1.0 - beta2) * g.array().square().matrix();
    Vector m_hat = m / (1.0 - std::pow(beta1, t));
    Vector v_hat = v / (1.0 - std::pow(beta2, t));
    p -= lr * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();
  };
  return Optimizer(update_m, update_v);
}

void Optimizer::update(Matrix &p, Matrix &m, Matrix &v, const Matrix &g,
                       Index t) const {
  update_matrix_(p, m, v, g, t);
}

void Optimizer::update(Vector &p, Vector &m, Vector &v, const Vector &g,
                       Index t) const {
  update_vector_(p, m, v, g, t);
}

} // namespace neural_network

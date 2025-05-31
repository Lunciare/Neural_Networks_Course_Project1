#include "Optimizer/Optimizer.h"
#include <cmath>

namespace neural_network {

namespace {

struct SGDCache {};

struct AdamCache {
  Matrix m_w, v_w;
  Vector m_b, v_b;
  Index t = 0;
};

} // namespace

Optimizer::Optimizer(MatrixUpdate mu, VectorUpdate vu, CacheInit ci)
    : update_matrix_(std::move(mu)), update_vector_(std::move(vu)),
      init_cache_(std::move(ci)) {}

Optimizer Optimizer::SGD(double lr) {
  return Optimizer([lr](Matrix &param, std::any &,
                        const Matrix &grad) { param -= lr * grad; },
                   [lr](Vector &param, std::any &, const Vector &grad) {
                     param -= lr * grad;
                   },
                   [](int, int) { return std::make_any<SGDCache>(); });
}

Optimizer Optimizer::Adam(double lr, double beta1, double beta2, double eps) {
  return Optimizer(
      [=](Matrix &param, std::any &cache_any, const Matrix &grad) {
        auto *cache = std::any_cast<AdamCache>(&cache_any);
        if (!cache)
          throw std::bad_any_cast();

        auto &m = cache->m_w, &v = cache->v_w;
        ++cache->t;

        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad.array().square().matrix();

        Matrix m_hat = m / (1.0 - std::pow(beta1, cache->t));
        Matrix v_hat = v / (1.0 - std::pow(beta2, cache->t));

        param -= lr * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();
      },
      [=](Vector &param, std::any &cache_any, const Vector &grad) {
        auto *cache = std::any_cast<AdamCache>(&cache_any);
        if (!cache)
          throw std::bad_any_cast();

        auto &m = cache->m_b, &v = cache->v_b;
        ++cache->t;

        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad.array().square().matrix();

        Vector m_hat = m / (1.0 - std::pow(beta1, cache->t));
        Vector v_hat = v / (1.0 - std::pow(beta2, cache->t));

        param -= lr * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();
      },
      [](int rows, int cols) {
        return std::make_any<AdamCache>(
            AdamCache{Matrix::Zero(rows, cols), Matrix::Zero(rows, cols),
                      Vector::Zero(rows), Vector::Zero(rows), 0});
      });
}

void Optimizer::update(Matrix &param, std::any &cache,
                       const Matrix &grad) const {
  update_matrix_(param, cache, grad);
}

void Optimizer::update(Vector &param, std::any &cache,
                       const Vector &grad) const {
  update_vector_(param, cache, grad);
}

std::any Optimizer::init_cache(int rows, int cols) const {
  return init_cache_(rows, cols);
}

} // namespace neural_network

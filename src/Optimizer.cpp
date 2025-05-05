#include "Optimizer.h"
#include <cassert>
#include <memory>

namespace neural_network {

Optimizer::Optimizer(MSignature optimizerA, VSignature optimizerB)
    : optimizerA_(std::move(optimizerA)), optimizerB_(std::move(optimizerB)) {}

Matrix &Optimizer::dMatrix() {
  static Matrix empty;
  return empty;
}

Vector &Optimizer::dVector() {
  static Vector empty;
  return empty;
}

Optimizer Optimizer::SGD(double learning_rate) {
  auto sgdUpdateA =
      [learning_rate](const Matrix &grad, const Matrix & /*currentWeights*/,
                      Matrix & /*memory*/, int /*time_step*/) -> Matrix {
    return learning_rate * grad;
  };

  auto sgdUpdateB =
      [learning_rate](const Vector &grad, const Vector & /*currentBiases*/,
                      Vector & /*memory*/, int /*time_step*/) -> Vector {
    return learning_rate * grad;
  };

  return Optimizer(sgdUpdateA, sgdUpdateB);
}

Optimizer Optimizer::Momentum(double learning_rate, double beta1) {
  auto momentum_update_a =
      [learning_rate, beta1](const Matrix &grad,
                             const Matrix & /*currentWeights*/, Matrix &memory,
                             Index /*time_step*/) -> Matrix {
    if (memory.size() == 0) {
      memory = Matrix::Zero(grad.rows(), grad.cols());
    }
    memory = beta1 * memory + (1 - beta1) * grad;
    return learning_rate * memory;
  };

  auto momentum_update_b =
      [learning_rate, beta1](const Vector &grad,
                             const Vector & /*currentBiases*/, Vector &memory,
                             Index /*time_step*/) -> Vector {
    if (memory.size() == 0) {
      memory = Vector::Zero(grad.size());
    }
    memory = beta1 * memory + (1 - beta1) * grad;
    return learning_rate * memory;
  };

  return Optimizer(momentum_update_a, momentum_update_b);
}

Optimizer Optimizer::Adam(double learning_rate, double beta1, double beta2,
                          double epsilon) {
  auto adamUpdateA = [learning_rate, beta1, beta2, epsilon](
                         const Matrix &grad, const Matrix & /*currentWeights*/,
                         Matrix &memory, Index time_step) -> Matrix {
    const Index rows = grad.rows();
    const Index cols = grad.cols();

    if (memory.size() == 0) {
      memory = Matrix::Zero(2 * rows, cols);
    }

    Matrix m = memory.topRows(rows);
    Matrix v = memory.bottomRows(rows);

    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * grad.array().square().matrix();

    memory.topRows(rows) = m;
    memory.bottomRows(rows) = v;

    const double beta1_t = std::pow(beta1, time_step);
    const double beta2_t = std::pow(beta2, time_step);

    Matrix mHat = m / (1 - beta1_t);
    Matrix vHat = v / (1 - beta2_t);

    return learning_rate *
           (mHat.array() / (vHat.array().sqrt() + epsilon)).matrix();
  };

  auto adamUpdateB = [learning_rate, beta1, beta2, epsilon](
                         const Vector &grad, const Vector & /*currentBiases*/,
                         Vector &memory, Index time_step) -> Vector {
    const Index size = grad.size();

    if (memory.size() == 0) {
      memory = Vector::Zero(2 * size);
    }

    Vector m = memory.head(size);
    Vector v = memory.tail(size);

    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * grad.array().square().matrix();

    memory.head(size) = m;
    memory.tail(size) = v;

    const double beta1_t = std::pow(beta1, time_step);
    const double beta2_t = std::pow(beta2, time_step);

    Vector mHat = m / (1 - beta1_t);
    Vector vHat = v / (1 - beta2_t);

    return learning_rate *
           (mHat.array() / (vHat.array().sqrt() + epsilon)).matrix();
  };

  return Optimizer(adamUpdateA, adamUpdateB);
}

Matrix Optimizer::getUpdateA(const Matrix &grad, const Matrix &cur_w,
                             Matrix &memory, Index time_step) const {
  assert(optimizerA_ && "Matrix optimizer not initialized");
  return optimizerA_(grad, cur_w, memory, time_step);
}

Vector Optimizer::getUpdateB(const Vector &grad, const Vector &cur_w,
                             Vector &memory, Index time_step) const {
  assert(optimizerB_ && "Vector optimizer not initialized");
  return optimizerB_(grad, cur_w, memory, time_step);
}

} // namespace neural_network
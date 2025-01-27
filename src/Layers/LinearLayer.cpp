#include "LinearLayer.h"

namespace NeuralNetwork {
LinearLayer::LinearLayer(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size),
      A_(output_size, input_size), b_(output_size) {
  initializeWeights();
}

Eigen::VectorXd LinearLayer::forward(const Eigen::VectorXd &X) const {
  return A_ * X + b_;
}

Eigen::VectorXd LinearLayer::backward(const Eigen::VectorXd &X,
                                      const Eigen::VectorXd &U, double h) {
  Eigen::MatrixXd grad_A = U * X.transpose();
  Eigen::VectorXd grad_b = U;

  A_ -= h * grad_A;
  b_ -= h * grad_b;

  Eigen::VectorXd new_U = A_.transpose() * U;

  return new_U;
}
} // namespace NeuralNetwork
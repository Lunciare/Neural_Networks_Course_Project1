#include "LossFunction.h"
#include <cassert>

namespace neural_network {

// Constructor that takes two function objects (loss and gradient functions)
LossFunction::LossFunction(Signature1 loss_func, Signature2 grad_func)
    : loss_func_(std::move(loss_func)), grad_func_(std::move(grad_func)) {}

/*******************************
 *  Loss Function Implementations
 *******************************/

// Euclidean distance loss (L2 norm)
LossFunction LossFunction::Euclid() {
  // Loss function implementation
  auto loss = [](const Matrix &x, const Matrix &y) {
    // Validate input dimensions
    assert(x.rows() == y.rows() && x.cols() == y.cols() &&
           "Input matrices must have the same dimensions");
    assert(x.cols() > 0 && "Number of samples must be greater than zero to "
                           "avoid division by zero");

    // Calculate mean squared error (0.5 * MSE for cleaner gradient)
    return 0.5 * (x - y).squaredNorm() / x.cols();
  };

  // Gradient implementation
  auto gradient = [](const Matrix &x, const Matrix &y) {
    assert(x.cols() > 0 && "Number of samples must be greater than zero to "
                           "avoid division by zero");

    // Derivative of MSE is simple difference, averaged over samples
    return (x - y) / x.cols();
  };

  return LossFunction(loss, gradient);
}

// Cross-entropy loss for classification (with epsilon for numerical stability)
LossFunction LossFunction::CrossEntropy() {
  constexpr float epsilon = 1e-8f; // Small constant to prevent log(0)

  auto loss = [epsilon](const Matrix &x, const Matrix &y) -> double {
    assert(x.cols() > 0 && "Must have at least one sample");

    // Cross-entropy formula: -Σ(y * log(x + ε)) / n
    return -(y.array() * (x.array() + epsilon).log()).sum() / x.cols();
  };

  auto gradient = [epsilon](const Matrix &x, const Matrix &y) -> Matrix {
    assert(x.cols() > 0 && "Must have at least one sample");

    // Gradient of cross-entropy: -y/(x + ε) / n
    return -(y.array() / (x.array() + epsilon)).matrix() / x.cols();
  };

  return LossFunction(loss, gradient);
}

// Cross-entropy loss that accepts logits (unnormalized scores)
LossFunction LossFunction::CrossEntropyWithLogits() {
  auto loss = [](const Matrix &x, const Matrix &y) -> double {
    assert(x.cols() > 0 && "Must have at least one sample");
    const int n = x.cols();
    double loss = 0.0;

    // Process each sample (column) separately
    for (int i = 0; i < n; ++i) {
      const auto x_col = x.col(i);
      const auto y_col = y.col(i);

      // Numerical stability: subtract maximum value before exponentiation
      const double max_coeff = x_col.maxCoeff();
      const Eigen::ArrayXd shifted = x_col.array() - max_coeff;

      // Calculate log-sum-exp
      const double sum_exp = shifted.exp().sum();
      const double log_sum_exp = std::log(sum_exp) + max_coeff;

      // Cross-entropy for this sample: -Σ(y * (x - logΣexp(x)))
      loss += -(y_col.array() * (x_col.array() - log_sum_exp)).sum();
    }

    return loss / n; // Average over all samples
  };

  auto gradient = [](const Matrix &x, const Matrix &y) -> Matrix {
    assert(x.cols() > 0 && "Must have at least one sample");
    const int n = x.cols();
    Matrix grad(x.rows(), x.cols());

    // Process each sample (column) separately
    for (int i = 0; i < n; ++i) {
      const auto x_col = x.col(i);
      const auto y_col = y.col(i);

      // Numerical stability: subtract maximum value before exponentiation
      const double max_coeff = x_col.maxCoeff();
      const Eigen::ArrayXd shifted = x_col.array() - max_coeff;

      // Softmax probabilities
      const Eigen::ArrayXd exp_shifted = shifted.exp();
      const double sum_exp = exp_shifted.sum();
      const Eigen::ArrayXd probs = exp_shifted / sum_exp;

      // Gradient for this sample: (probs - y) / n
      grad.col(i) = ((probs - y_col.array()) / static_cast<double>(n)).matrix();
    }

    return grad;
  };

  return LossFunction(loss, gradient);
}

/*******************************
 *  Public Interface Methods
 *******************************/

// Calculate loss between predicted (x) and target (y) values
double LossFunction::computeLoss(const Matrix &x, const Matrix &y) const {
  assert(loss_func_ && "Loss function not initialized");
  return loss_func_(x, y);
}

// Calculate gradient of loss with respect to predicted values (x)
Matrix LossFunction::computeGradient(const Matrix &x, const Matrix &y) const {
  assert(grad_func_ && "Gradient function not initialized");
  return grad_func_(x, y);
}

} // namespace neural_network
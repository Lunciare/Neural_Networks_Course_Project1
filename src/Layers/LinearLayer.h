#pragma once

#include <Eigen/Dense>
#include <random>

namespace NeuralNetwork {
class LinearLayer {
private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  size_t input_size;
  size_t output_size;

  void initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    for (size_t i = 0; i < A_.rows(); ++i) {
      for (size_t j = 0; j < A_.cols(); ++j) {
        A_(i, j) = dist(gen);
      }
    }

    for (size_t i = 0; i < b_.size(); ++i) {
      b_(i) = dist(gen);
    }
  }

public:
  LinearLayer(size_t input_size, size_t output_size);

  Eigen::VectorXd forward(const Eigen::VectorXd &X) const;

  Eigen::VectorXd backward(const Eigen::VectorXd &X, const Eigen::VectorXd &U,
                           double h);
};
} // namespace NeuralNetwork
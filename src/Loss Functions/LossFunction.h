#pragma once
#include "eigen/Eigen/Dense"

namespace neural_network
{

// Provides static methods for loss computation (e.g. mean squared error).
class LossFunction
{
public:
    // Computes the Mean Squared Error (MSE) between prediction and target
    // vectors.
    static double mse(const Eigen::VectorXd& y_pred,
                      const Eigen::VectorXd& y_true);

    // Computes the gradient of MSE with respect to y_pred.
    static Eigen::VectorXd mseGrad(const Eigen::VectorXd& y_pred,
                                   const Eigen::VectorXd& y_true);
};

}// namespace neural_network

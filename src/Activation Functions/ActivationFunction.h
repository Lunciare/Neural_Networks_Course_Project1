#pragma once
#include "eigen/Eigen/Dense"

namespace neural_network
{

// Provides static methods for activation functions and their derivatives.
class ActivationFunction
{
public:
    enum class Type
    {
        ReLU,
        Sigmoid,
        Identity,
        Tanh
    };

    // Applies the activation function (elementwise) to the input vector.
    static Eigen::VectorXd apply(Type type, const Eigen::VectorXd& x);

    // Computes the derivative of the activation function (elementwise), given the
    // activation output.
    static Eigen::VectorXd derivative(Type type, const Eigen::VectorXd& y);
};

}// namespace neural_network

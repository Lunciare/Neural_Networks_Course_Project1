#include "ActivationFunction.h"
#include <cassert>
#include <cmath>

namespace neural_network
{

Eigen::VectorXd ActivationFunction::apply(Type type, const Eigen::VectorXd& x)
{
    switch (type)
    {
        case Type::ReLU:
            return x.array().max(0.0).matrix();
        case Type::Sigmoid:
            return (1.0 / (1.0 + (-x.array()).exp())).matrix();
        case Type::Identity:
            return x;
        case Type::Tanh:
            return x.array().tanh().matrix();
        default:
            assert(false && "Unknown activation type");
            return x;
    }
}

Eigen::VectorXd ActivationFunction::derivative(Type type,
                                               const Eigen::VectorXd& y)
{
    switch (type)
    {
        case Type::ReLU:
            return (y.array() > 0.0).cast<double>().matrix();
        case Type::Sigmoid:
            return (y.array() * (1.0 - y.array())).matrix();
        case Type::Identity:
            return Eigen::VectorXd::Ones(y.size());
        case Type::Tanh:
            return (1.0 - y.array().square()).matrix();
        default:
            assert(false && "Unknown activation type");
            return y;
    }
}

}// namespace neural_network

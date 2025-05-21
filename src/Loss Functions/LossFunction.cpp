#include "LossFunction.h"
#include <cassert>

namespace neural_network
{

double LossFunction::mse(const Eigen::VectorXd& y_pred,
                         const Eigen::VectorXd& y_true)
{
    assert(y_pred.size() == y_true.size());
    auto diff = y_pred - y_true;
    return diff.squaredNorm() / diff.size();
}

Eigen::VectorXd LossFunction::mseGrad(const Eigen::VectorXd& y_pred,
                                      const Eigen::VectorXd& y_true)
{
    assert(y_pred.size() == y_true.size());
    return (2.0 / y_pred.size()) * (y_pred - y_true);
}

}// namespace neural_network

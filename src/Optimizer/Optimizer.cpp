#include "Optimizer.h"
#include <cmath>

namespace neural_network
{

Optimizer::Optimizer(double lr)
    : type_(OptimizerType::SGD), lr_(lr), beta1_(0), beta2_(0), eps_(0) {}

Optimizer::Optimizer(double lr, double beta1, double beta2, double eps)
    : type_(OptimizerType::Adam), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps) {}

void Optimizer::update(Eigen::MatrixXd& param, Eigen::MatrixXd& m,
                       Eigen::MatrixXd& v, const Eigen::MatrixXd& grad,
                       size_t t) const
{
    if (type_ == OptimizerType::SGD)
    {
        param -= lr_ * grad;
    }
    else if (type_ == OptimizerType::Adam)
    {
        m = beta1_ * m + (1.0 - beta1_) * grad;
        v = beta2_ * v + (1.0 - beta2_) * grad.array().square().matrix();
        Eigen::MatrixXd m_hat = m / (1.0 - std::pow(beta1_, t));
        Eigen::MatrixXd v_hat = v / (1.0 - std::pow(beta2_, t));
        param -= (lr_ * m_hat.array() / (v_hat.array().sqrt() + eps_)).matrix();
    }
}

void Optimizer::update(Eigen::VectorXd& param, Eigen::VectorXd& m,
                       Eigen::VectorXd& v, const Eigen::VectorXd& grad,
                       size_t t) const
{
    if (type_ == OptimizerType::SGD)
    {
        param -= lr_ * grad;
    }
    else if (type_ == OptimizerType::Adam)
    {
        m = beta1_ * m + (1.0 - beta1_) * grad;
        v = beta2_ * v + (1.0 - beta2_) * grad.array().square().matrix();
        Eigen::VectorXd m_hat = m / (1.0 - std::pow(beta1_, t));
        Eigen::VectorXd v_hat = v / (1.0 - std::pow(beta2_, t));
        param -= (lr_ * m_hat.array() / (v_hat.array().sqrt() + eps_)).matrix();
    }
}

}// namespace neural_network

#pragma once
#include "eigen/Eigen/Dense"
#include <cstddef>

namespace neural_network
{

enum class OptimizerType
{
    SGD,
    Adam
};

class Optimizer
{
public:
    // Constructor for SGD
    explicit Optimizer(double lr);

    // Constructor for Adam
    Optimizer(double lr, double beta1, double beta2, double eps);

    void setType(OptimizerType type) { type_ = type; }
    OptimizerType type() const { return type_; }

    // Updates for matrix and vector parameters
    void update(Eigen::MatrixXd& param, Eigen::MatrixXd& m, Eigen::MatrixXd& v,
                const Eigen::MatrixXd& grad, size_t t) const;
    void update(Eigen::VectorXd& param, Eigen::VectorXd& m, Eigen::VectorXd& v,
                const Eigen::VectorXd& grad, size_t t) const;

    double lr() const { return lr_; }
    double beta1() const { return beta1_; }
    double beta2() const { return beta2_; }
    double eps() const { return eps_; }

private:
    OptimizerType type_;
    double lr_;
    double beta1_, beta2_, eps_;
};

}// namespace neural_network

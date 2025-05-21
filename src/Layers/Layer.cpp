#include "Layer.h"
#include <cassert>
#include <cmath>
#include <fstream>

namespace neural_network
{

Eigen::MatrixXd Layer::initWeights(size_t out, size_t in)
{
    double limit = std::sqrt(6.0 / (out + in));
    Eigen::MatrixXd w = Eigen::MatrixXd::Random(out, in);
    w *= limit;
    return w;
}
Eigen::VectorXd Layer::initBiases(size_t out)
{
    return Eigen::VectorXd::Zero(out);
}

Layer::Layer(size_t input_size, size_t output_size,
             ActivationFunction::Type activation, const Optimizer& optimizer)
    : input_size_(input_size), output_size_(output_size),
      activation_type_(activation),
      weights_(initWeights(output_size, input_size)),
      biases_(initBiases(output_size)),
      m_w_(Eigen::MatrixXd::Zero(output_size, input_size)),
      v_w_(Eigen::MatrixXd::Zero(output_size, input_size)),
      m_b_(Eigen::VectorXd::Zero(output_size)),
      v_b_(Eigen::VectorXd::Zero(output_size)), t_(0),
      last_input_(Eigen::VectorXd::Zero(input_size)),
      last_z_(Eigen::VectorXd::Zero(output_size)), optimizer_(optimizer) {}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input)
{
    assert(input.size() == input_size_);
    last_input_ = input;
    last_z_ = weights_ * input + biases_;
    return ActivationFunction::apply(activation_type_, last_z_);
}

Eigen::VectorXd Layer::backward(const Eigen::VectorXd& grad_output)
{
    assert(grad_output.size() == output_size_);
    ++t_;
    Eigen::VectorXd act_deriv = ActivationFunction::derivative(
            activation_type_, ActivationFunction::apply(activation_type_, last_z_));
    Eigen::VectorXd dz = grad_output.array() * act_deriv.array();

    Eigen::MatrixXd grad_w = dz * last_input_.transpose();
    Eigen::VectorXd grad_b = dz;

    optimizer_.update(weights_, m_w_, v_w_, grad_w, t_);
    optimizer_.update(biases_, m_b_, v_b_, grad_b, t_);

    return weights_.transpose() * dz;
}

bool Layer::saveWeights(const std::string& filename) const
{
    std::ofstream out(filename);
    if (!out.is_open())
        return false;
    out << output_size_ << " " << input_size_ << "\n";
    for (size_t i = 0; i < output_size_; ++i)
    {
        for (size_t j = 0; j < input_size_; ++j)
            out << weights_(i, j) << " ";
        out << "\n";
    }
    for (size_t i = 0; i < output_size_; ++i)
        out << biases_(i) << " ";
    out << "\n";
    return true;
}

bool Layer::loadWeights(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in.is_open())
        return false;
    size_t out, in_size;
    in >> out >> in_size;
    if (out != output_size_ || in_size != input_size_)
        return false;
    for (size_t i = 0; i < output_size_; ++i)
        for (size_t j = 0; j < input_size_; ++j)
            in >> weights_(i, j);
    for (size_t i = 0; i < output_size_; ++i)
        in >> biases_(i);
    m_w_.setZero();
    v_w_.setZero();
    m_b_.setZero();
    v_b_.setZero();
    t_ = 0;
    return true;
}

}// namespace neural_network

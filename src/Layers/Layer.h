#pragma once
#include "ActivationFunction.h"
#include "Optimizer.h"
#include "eigen/Eigen/Dense"
#include <string>

namespace neural_network
{

// Fully-connected (dense) neural network layer with activation and optimizer.
class Layer
{
public:
    Layer(size_t input_size, size_t output_size,
          ActivationFunction::Type activation, const Optimizer& optimizer);

    Eigen::VectorXd forward(const Eigen::VectorXd& input);

    // Backward pass: grad_output = gradient w.r.t. output. Returns gradient
    // w.r.t. input.
    Eigen::VectorXd backward(const Eigen::VectorXd& grad_output);

    bool saveWeights(const std::string& filename) const;
    bool loadWeights(const std::string& filename);

    size_t inputSize() const { return input_size_; }
    size_t outputSize() const { return output_size_; }

private:
    size_t input_size_, output_size_;
    ActivationFunction::Type activation_type_;
    Eigen::MatrixXd weights_;
    Eigen::VectorXd biases_;

    Eigen::MatrixXd m_w_, v_w_;
    Eigen::VectorXd m_b_, v_b_;
    size_t t_;

    Eigen::VectorXd last_input_;
    Eigen::VectorXd last_z_;

    Optimizer optimizer_;// stored by value!

    static Eigen::MatrixXd initWeights(size_t out, size_t in);
    static Eigen::VectorXd initBiases(size_t out);
};

}// namespace neural_network

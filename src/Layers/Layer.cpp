#include "Layers/Layer.h"
#include "Utilities/Random.h"

#include <cassert>
#include <cmath>
#include <fstream>

namespace neural_network {

Matrix Layer::initWeights(Index out, Index in) {
  double stddev = std::sqrt(2.0 / (in + out));
  return Random::global().normalMatrix(out, in, 0.0, stddev);
}

Vector Layer::initBiases(Index out) { return Vector::Zero(out); }

Layer::Layer(Index in, Index out, ActivationFunction activation)
    : weights_(initWeights(out, in)), biases_(initBiases(out)),
      m_w_(Matrix::Zero(out, in)), v_w_(Matrix::Zero(out, in)),
      m_b_(Vector::Zero(out)), v_b_(Vector::Zero(out)), t_(0),
      last_input_(Vector::Zero(in)), last_z_(Vector::Zero(out)),
      optimizer_(nullptr), activation_(std::move(activation)) {}

Vector Layer::forward(const Vector &input) const {
  Vector z = weights_ * input + biases_;
  return activation_.apply(activation_type_, z);
}

Vector Layer::forwardTrain(const Vector &input) {
  last_input_ = input;
  last_z_ = weights_ * input + biases_;
  return activation_.apply(activation_type_, last_z_);
}

Vector Layer::backward(const Vector &grad_output) {
  ++t_;

  Vector deriv = activation_.derivative(activation_type_, last_z_);
  Vector dz = grad_output.array() * deriv.array();

  Matrix grad_w = dz * last_input_.transpose();
  Vector grad_b = dz;

  if (optimizer_) {
    optimizer_->update(weights_, m_w_, v_w_, grad_w, t_);
    optimizer_->update(biases_, m_b_, v_b_, grad_b, t_);
  }

  return weights_.transpose() * dz;
}

void Layer::setOptimizer(Optimizer *optimizer) { optimizer_ = optimizer; }

Layer::IOStatus Layer::saveWeights(const std::string &filename) const {
  std::ofstream out(filename);
  if (!out.is_open())
    return IOStatus::IOError;

  out << weights_.rows() << ' ' << weights_.cols() << '\n';
  for (Index i = 0; i < weights_.rows(); ++i) {
    for (Index j = 0; j < weights_.cols(); ++j)
      out << weights_(i, j) << ' ';
    out << '\n';
  }

  out << biases_.size() << '\n';
  for (Index i = 0; i < biases_.size(); ++i)
    out << biases_(i) << ' ';
  out << '\n';

  return out.good() ? IOStatus::OK : IOStatus::IOError;
}

Layer::IOStatus Layer::loadWeights(const std::string &filename) {
  std::ifstream in(filename);
  if (!in.is_open())
    return IOStatus::IOError;

  Index rows, cols;
  in >> rows >> cols;
  weights_.resize(rows, cols);
  for (Index i = 0; i < rows; ++i)
    for (Index j = 0; j < cols; ++j)
      in >> weights_(i, j);

  Index bsize;
  in >> bsize;
  biases_.resize(bsize);
  for (Index i = 0; i < bsize; ++i)
    in >> biases_(i);

  return in.good() ? IOStatus::OK : IOStatus::IOError;
}

} // namespace neural_network

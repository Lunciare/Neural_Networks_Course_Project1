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

Layer::Layer(Index in, Index out, ActivationFunction::Type activation,
             const Optimizer &optimizer)
    : input_size_(in), output_size_(out), activation_type_(activation),
      weights_(initWeights(out, in)), biases_(initBiases(out)),
      m_w_(Matrix::Zero(out, in)), v_w_(Matrix::Zero(out, in)),
      m_b_(Vector::Zero(out)), v_b_(Vector::Zero(out)), t_(0),
      last_input_(Vector::Zero(in)), last_z_(Vector::Zero(out)),
      optimizer_(optimizer) {}

Vector Layer::forward(const Vector &input) {
  assert(input.size() == input_size_);
  last_input_ = input;
  last_z_ = weights_ * input + biases_;
  return ActivationFunction::apply(activation_type_, last_z_);
}

Vector Layer::backward(const Vector &grad_output) {
  assert(grad_output.size() == output_size_);
  ++t_;

  Vector activated = ActivationFunction::apply(activation_type_, last_z_);
  auto deriv = ActivationFunction::derivative(activation_type_, activated);

  auto dz = grad_output.array() * deriv.array();

  auto grad_w = dz.matrix() * last_input_.transpose();
  auto grad_b = dz.matrix();

  optimizer_.update(weights_, m_w_, v_w_, grad_w, t_);
  optimizer_.update(biases_, m_b_, v_b_, grad_b, t_);

  // propagate gradient backward
  return weights_.transpose() * dz.matrix();
}

bool Layer::saveWeights(const std::string &filename) const {
  std::ofstream out(filename);
  if (!out.is_open())
    return false;

  out << weights_.rows() << ' ' << weights_.cols() << '\n';
  for (Index i = 0; i < weights_.rows(); ++i) {
    for (Index j = 0; j < weights_.cols(); ++j) {
      out << weights_(i, j) << ' ';
    }
    out << '\n';
  }

  out << biases_.size() << '\n';
  for (Index i = 0; i < biases_.size(); ++i) {
    out << biases_(i) << ' ';
  }
  out << '\n';

  return out.good();
}

bool Layer::loadWeights(const std::string &filename) {
  std::ifstream in(filename);
  if (!in.is_open())
    return false;

  Index rows, cols;
  in >> rows >> cols;
  weights_.resize(rows, cols);
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      in >> weights_(i, j);
    }
  }

  Index bsize;
  in >> bsize;
  biases_.resize(bsize);
  for (Index i = 0; i < bsize; ++i) {
    in >> biases_(i);
  }

  return in.good();
}

} // namespace neural_network

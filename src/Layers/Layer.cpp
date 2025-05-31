#include "Layers/Layer.h"
#include "Utilities/FileReader.h"
#include "Utilities/FileWriter.h"
#include "Utilities/Random.h"

#include <cassert>
#include <cmath>
#include <fstream>

namespace neural_network {

Matrix Layer::initWeights(Out out, In in) {
  double stddev = std::sqrt(2.0 / (in + out));
  return Random::global().normalMatrix(out, in, 0.0, stddev);
}

Vector Layer::initBiases(Out out) { return Vector::Zero(out); }

Layer::Layer(In in, Out out, ActivationFunction activation)
    : weights_(initWeights(out, in)), biases_(initBiases(out)),
      m_w_(Matrix::Zero(out, in)), v_w_(Matrix::Zero(out, in)),
      m_b_(Vector::Zero(out)), v_b_(Vector::Zero(out)), t_(0),
      last_input_(Vector::Zero(in)), last_z_(Vector::Zero(out)),
      optimizer_(nullptr), activation_(std::move(activation)) {}

Vector Layer::forward(const Vector &input) {
  assert(input.size() == weights_.cols());
  last_input_ = input;
  last_z_ = weights_ * input + biases_;
  return activation_.apply(activation_type_, last_z_);
}

Vector Layer::predict(const Vector &input) const {
  assert(input.size() == weights_.cols());
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

template <class Reader> void Layer::read(Reader &in) {
  int activation_int;
  in >> activation_int;
  activation_type_ = static_cast<ActivationFunction::Type>(activation_int);
  activation_ = ActivationFunction(); // simple re-init, no params

  in >> weights_;
  in >> biases_;

  // reset moments and time step
  m_w_ = Matrix::Zero(weights_.rows(), weights_.cols());
  v_w_ = Matrix::Zero(weights_.rows(), weights_.cols());
  m_b_ = Vector::Zero(biases_.size());
  v_b_ = Vector::Zero(biases_.size());
  t_ = 0;

  last_input_ = Vector::Zero(weights_.cols());
  last_z_ = Vector::Zero(weights_.rows());
}

template <class Writer> void Layer::write(Writer &out) const {
  out << static_cast<int>(activation_type_);
  out << weights_;
  out << biases_;
}

// Explicit instantiations
template void Layer::read(FileReader &);
template void Layer::write(FileWriter &) const;

} // namespace neural_network

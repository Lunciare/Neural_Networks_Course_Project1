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
      last_input_(Vector::Zero(in)), last_z_(Vector::Zero(out)),
      activation_(std::move(activation)) {}

Vector Layer::forward(const Vector &input) {
  assert(input.size() == weights_.cols());
  last_input_ = input;
  last_z_ = weights_ * input + biases_;
  return activation_.apply(last_z_);
}

Vector Layer::predict(const Vector &input) const {
  assert(input.size() == weights_.cols());
  Vector z = weights_ * input + biases_;
  return activation_.apply(z);
}

Vector Layer::forwardTrain(const Vector &input) {
  last_input_ = input;
  last_z_ = weights_ * input + biases_;
  return activation_.apply(last_z_);
}

Vector Layer::backward(const Vector &grad_output, const Optimizer &optimizer) {
  if (!cache_.has_value()) {
    throw std::runtime_error("Optimizer cache not initialized");
  }

  Vector deriv = activation_.derivative(last_z_);
  Vector dz = grad_output.array() * deriv.array();

  Matrix grad_w = dz * last_input_.transpose();
  Vector grad_b = dz;

  optimizer.update(weights_, cache_, grad_w);
  optimizer.update(biases_, cache_, grad_b);

  return weights_.transpose() * dz;
}

void Layer::set_cache(const Optimizer &opt) {
  cache_ = opt.init_cache(weights_.rows(), weights_.cols());
}

void Layer::free_cache() { cache_.reset(); }

template <class Reader> void Layer::read(Reader &in) {
  int af_code;
  in >> af_code;
  activation_type_ = static_cast<ActivationFunction::Type>(af_code);
  activation_ = ActivationFunction::create(activation_type_);

  in >> weights_;
  in >> biases_;

  last_input_ = Vector::Zero(weights_.cols());
  last_z_ = Vector::Zero(weights_.rows());

  cache_.reset();
}

template <class Writer> void Layer::write(Writer &out) const {
  out << static_cast<int>(activation_type_);
  out << weights_;
  out << biases_;
}

// explicit instantiations
template void Layer::read(FileReader &);
template void Layer::write(FileWriter &) const;

} // namespace neural_network

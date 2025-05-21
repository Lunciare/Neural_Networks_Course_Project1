#include "Layers/Layer.h"
#include "Utilities/Random.h"
#include <cassert>
#include <cmath>
#include <fstream>

namespace neural_network {

Matrix Layer::initWeights(Out out, In in) {
  double stddev =
      std::sqrt(2.0 / (static_cast<Index>(in) + static_cast<Index>(out)));
  // Use EigenRand to initialize weights with normal distribution
  return Eigen::Rand::normal<Matrix>(out, in, Random::global().engine(), 0.0,
                                     stddev);
}
Vector Layer::initBiases(Out out) {
  return Vector::Zero(static_cast<Index>(out));
}

Layer::Layer(In in, Out out, ActivationFunction::Type activation,
             const Optimizer &optimizer)
    : input_size_(static_cast<Index>(in)),
      output_size_(static_cast<Index>(out)), activation_type_(activation),
      weights_(initWeights(out, in)), biases_(initBiases(out)),
      m_w_(Matrix::Zero(static_cast<Index>(out), static_cast<Index>(in))),
      v_w_(Matrix::Zero(static_cast<Index>(out), static_cast<Index>(in))),
      m_b_(Vector::Zero(static_cast<Index>(out))),
      v_b_(Vector::Zero(static_cast<Index>(out))), t_(0),
      last_input_(Vector::Zero(static_cast<Index>(in))),
      last_z_(Vector::Zero(static_cast<Index>(out))), optimizer_(optimizer) {}

Vector Layer::forward(const Vector &input) {
  assert(input.size() == input_size_);
  last_input_ = input;
  last_z_ = weights_ * input + biases_;
  return ActivationFunction::apply(activation_type_, last_z_);
}

Vector Layer::backward(const Vector &grad_output) {
  assert(grad_output.size() == output_size_);
  ++t_;
  auto act_deriv = ActivationFunction::derivative(
      activation_type_, ActivationFunction::apply(activation_type_, last_z_));
  auto dz = grad_output.array() * act_deriv.array();
  auto grad_w = dz.matrix() * last_input_.transpose();
  auto grad_b = dz.matrix();

  optimizer_.update(weights_, m_w_, v_w_, grad_w, t_);
  optimizer_.update(biases_, m_b_, v_b_, grad_b, t_);

  return weights_.transpose() * dz.matrix();
}

bool Layer::saveWeights(const std::string &filename) const {
  std::ofstream out(filename, std::ios::binary);
  if (!out)
    return false;
  // Сохраняем размеры
  Index rows = weights_.rows();
  Index cols = weights_.cols();
  out.write((char *)&rows, sizeof(rows));
  out.write((char *)&cols, sizeof(cols));
  // Сохраняем веса (построчно)
  for (Index i = 0; i < rows; ++i)
    out.write(reinterpret_cast<const char *>(weights_.data() + i * cols),
              sizeof(double) * cols);
  // Сохраняем размер bias
  Index bsize = biases_.size();
  out.write((char *)&bsize, sizeof(bsize));
  // Сохраняем biases
  for (Index i = 0; i < bsize; ++i)
    out.write((char *)&biases_(i), sizeof(double));
  return (bool)out;
}

bool Layer::loadWeights(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in)
    return false;
  // Читаем размеры
  Index rows, cols;
  in.read((char *)&rows, sizeof(rows));
  in.read((char *)&cols, sizeof(cols));
  weights_.resize(rows, cols);
  for (Index i = 0; i < rows; ++i)
    in.read(reinterpret_cast<char *>(weights_.data() + i * cols),
            sizeof(double) * cols);
  // Читаем размер bias
  Index bsize;
  in.read((char *)&bsize, sizeof(bsize));
  biases_.resize(bsize);
  for (Index i = 0; i < bsize; ++i)
    in.read((char *)&biases_(i), sizeof(double));
  return (bool)in;
}

} // namespace neural_network

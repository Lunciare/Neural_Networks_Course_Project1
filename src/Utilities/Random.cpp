#include "Utilities/Random.h"

namespace neural_network {

Random::Random(std::uint64_t seed) : generator_(seed) {}

Matrix Random::uniformMatrix(Index rows, Index cols, double a, double b) {
  // Используй generator_ как источник случайных чисел
  return Eigen::Rand::uniformReal<Matrix>(rows, cols, generator_, a, b);
}

Vector Random::uniformVector(Index size, double a, double b) {
  // Vector всегда делай как (size, 1) и потом, если надо, .col(0)
  return Eigen::Rand::uniformReal<Matrix>(size, 1, generator_, a, b);
}

Matrix Random::normalMatrix(Index rows, Index cols, double mean,
                            double stddev) {
  return Eigen::Rand::normal<Matrix>(rows, cols, generator_, mean, stddev);
}

Vector Random::normalVector(Index size, double mean, double stddev) {
  return Eigen::Rand::normal<Matrix>(size, 1, generator_, mean, stddev);
}

// Реализация глобального генератора (singleton)
Random &Random::global() {
  static Random instance(k_default_seed_);
  return instance;
}

} // namespace neural_network

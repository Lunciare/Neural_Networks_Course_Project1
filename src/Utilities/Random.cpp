#include "Utilities/Random.h"

namespace neural_network {

Random::Random(std::uint64_t seed) : generator_(seed) {}

Matrix Random::uniformMatrix(Index rows, Index cols, double a, double b) {
  return Eigen::Rand::uniformReal<Matrix>(rows, cols, generator_, a, b);
}

Vector Random::uniformVector(Index size, double a, double b) {
  return Eigen::Rand::uniformReal<Matrix>(size, 1, generator_, a, b);
}

Matrix Random::normalMatrix(Index rows, Index cols, double mean,
                            double stddev) {
  return Eigen::Rand::normal<Matrix>(rows, cols, generator_, mean, stddev);
}

Vector Random::normalVector(Index size, double mean, double stddev) {
  return Eigen::Rand::normal<Matrix>(size, 1, generator_, mean, stddev);
}

Random &Random::global() {
  static Random instance(k_default_seed_);
  return instance;
}

} // namespace neural_network

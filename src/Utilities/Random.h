#pragma once
#include "Utilities/LinAlg.h"

namespace neural_network {

class Random {
public:
  explicit Random(std::uint64_t seed);

  Matrix uniformMatrix(Index rows, Index cols, double a, double b);
  Vector uniformVector(Index size, double a, double b);

  Matrix normalMatrix(Index rows, Index cols, double mean, double stddev);
  Vector normalVector(Index size, double mean, double stddev);

  static Random &global();

  Eigen::Rand::Vmt19937_64 &engine() { return generator_; }

private:
  static constexpr std::uint64_t k_default_seed_ = 42;
  Eigen::Rand::Vmt19937_64 generator_;
};

} // namespace neural_network

#pragma once

#include <EigenRand/EigenRand>
#include <random>

#include "Math.h"

namespace neural_network {
class Random {
  static constexpr Index k_default_seed = 42241;

public:
  static Matrix generateNormalMatrix(Index rows, Index cols);
  static Matrix generateBernoulliMatrix(Index rows, Index cols,
                                        double success_prob);

private:
  static thread_local std::mt19937 generator_;
};

} // namespace neural_network

#pragma once

#include "ActivationFunctions/ActivationFunction.h"
#include "Optimizer/Optimizer.h"
#include "Utilities/Utils.h"

#include <any>
#include <optional>
#include <string>
#include <vector>

namespace neural_network {

class Layer {
public:
  Layer();
  Layer(In in, Out out, ActivationFunction activation);

  Vector forward(const Vector &input);
  Vector predict(const Vector &input) const;

  Vector backward(const Vector &grad_output, const Optimizer &optimizer);

  void setCache(const Optimizer &opt);
  void freeCache();

  template <class Reader> void read(Reader &in);
  template <class Writer> void write(Writer &out) const;

private:
  static Matrix initWeights(Out out, In in);
  static Vector initBiases(Out out);

  ActivationFunction::Type activation_type_;
  ActivationFunction activation_;

  Matrix weights_;
  Vector biases_;

  std::any cache_; // cache from Optimizer

  // Caches for backprop
  Vector last_input_;
  Vector last_z_;
};

} // namespace neural_network

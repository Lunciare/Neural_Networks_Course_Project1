#pragma once

#include "ActivationFunctions/ActivationFunction.h"
#include "Optimizer/Optimizer.h"
#include "Utilities/LinAlg.h"

#include <string>

namespace neural_network {

class Layer {
public:
  Layer(Index in, Index out, ActivationFunction::Type activation,
        const Optimizer &optimizer);

  Vector forward(const Vector &input);

  Vector backward(const Vector &grad_output);

  bool saveWeights(const std::string &filename) const;

  bool loadWeights(const std::string &filename);

private:
  static Matrix initWeights(Index out, Index in);

  static Vector initBiases(Index out);

  Index input_size_;
  Index output_size_;
  ActivationFunction::Type activation_type_;

  Matrix weights_;
  Vector biases_;

  // Adam optimizer moments
  Matrix m_w_;
  Matrix v_w_;
  Vector m_b_;
  Vector v_b_;
  Index t_;

  // Caches for backprop
  Vector last_input_;
  Vector last_z_;

  Optimizer optimizer_;
};

} // namespace neural_network

#pragma once

#include "ActivationFunctions/ActivationFunction.h"
#include "Optimizer/Optimizer.h"
#include "Utilities/LinAlg.h"

namespace neural_network {

class Layer {
public:
  Layer(In in, Out out, ActivationFunction::Type activation,
        const Optimizer &optimizer);

  // Forward pass (input: column vector)
  Vector forward(const Vector &input);

  // Backward pass (grad_output: column vector)
  Vector backward(const Vector &grad_output);

  bool saveWeights(const std::string &filename) const;
  bool loadWeights(const std::string &filename);

private:
  Index input_size_;
  Index output_size_;
  ActivationFunction::Type activation_type_;
  Matrix weights_;
  Vector biases_;

  Matrix m_w_, v_w_;
  Vector m_b_, v_b_;
  Index t_;

  Vector last_input_;
  Vector last_z_;

  Optimizer optimizer_;
  static Matrix initWeights(Out out, In in);
  static Vector initBiases(Out out);
};

} // namespace neural_network

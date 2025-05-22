#pragma once

#include "ActivationFunctions/ActivationFunction.h"
#include "Optimizer/Optimizer.h"
#include "Utilities/LinAlg.h"

#include <string>

namespace neural_network {

// A fully-connected neural network layer.
class Layer {
public:
  // Constructs a layer with the given input size, output size,
  // activation function type and shared optimizer.
  Layer(Index in, Index out, ActivationFunction::Type activation,
        const Optimizer &optimizer);

  // Runs forward propagation through this layer.
  Vector forward(const Vector &input);

  // Runs backward propagation, updates parameters, and returns
  // the gradient for the previous layer.
  Vector backward(const Vector &grad_output);

  // Saves weights and biases in a human-readable text format.
  bool saveWeights(const std::string &filename) const;

  // Loads weights and biases from a human-readable text format.
  bool loadWeights(const std::string &filename);

private:
  // Initializer for weights.
  static Matrix initWeights(Index out, Index in);

  // Zero initializer for biases.
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
